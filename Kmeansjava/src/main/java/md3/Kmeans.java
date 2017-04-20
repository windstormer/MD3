package md3;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.*;
import java.lang.*;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hdfs.server.namenode.INodesInPath;
import java.io.InputStreamReader;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.util.GenericOptionsParser;

import com.google.protobuf.ByteString.Output;
import com.sun.tools.javac.util.StringUtils;

public class Kmeans {

	public static class WClusterMapper extends Mapper<Object, Text, Text, Text> {

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String centroid = context.getConfiguration().get("centroid");
			int dis = Integer.parseInt(context.getConfiguration().get("dis"));
			String[] itr = centroid.split(",");
			Vector<ArrayList<Double>> cen = new Vector<ArrayList<Double>>();
			for (int i = 0; i < itr.length; i++) {
				
				cen.add(i, new ArrayList<Double>());
				String[] dimension = itr[i].split(" ");
				for (int j = 0; j < dimension.length; j++) {
					cen.get(i).add(j, Double.parseDouble(dimension[j]));
				}
			}
			// for(int i=0;i<cen.size();i++)
			// {
			// 	StringBuilder builder = new StringBuilder();
			// for (int j=0;j<cen.get(i).size();j++) {
			// 	builder.append(cen.get(i).get(j).toString() + ",");
			// }
			// 	context.write(new Text(String.valueOf(i)),new Text(builder.toString().substring(0, builder.toString().length() - 1)));
			// }

			String[] Data = value.toString().split(" ");
			Double mindis = Double.MAX_VALUE;
			
			int clus = 0;
			for (int j = 0; j < cen.size(); j++) {
				Double sum = 0.0;
				for (int i = 0; i < Data.length; i++) {
					if(dis==0)
						sum += Math.pow(Double.parseDouble(Data[i]) - cen.get(j).get(i).doubleValue(),2);
					else
						sum += Math.abs(Double.parseDouble(Data[i]) - cen.get(j).get(i).doubleValue());
				}
				if (sum < mindis) {
					mindis = sum;
					clus = j;
				}
			}
			context.write(new Text("cost"), new Text(mindis.toString()));
			StringBuilder builder = new StringBuilder();
			for (String s : Data) {
				builder.append(s + ",");
			}
			context.write(new Text(String.valueOf(clus)),
				new Text(builder.toString().substring(0, builder.toString().length() - 1)));
			// context.write(new Text(String.valueOf(clus)), new Text(String.valueOf(mindis)));
		}
	}

	public static class CentroidReducer extends Reducer<Text, Text, Text, Text> {
		private MultipleOutputs mos;

		public void setup(Context context) {
			mos = new MultipleOutputs(context);
		}

		public void cleanup(Context context) throws IOException, InterruptedException {
			mos.close();
		}

		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			if (key.toString().equals("cost")) {
				double totalcost = 0.0;
				for (Text val : values) {
					totalcost += Double.parseDouble(val.toString());
				}
				mos.write("cost", new Text("cost"), new Text(String.valueOf(totalcost)), "cost");
			} else {
				Vector<ArrayList<Double>> clusdata = new Vector<ArrayList<Double>>();
				for (Text val : values) {
					ArrayList<Double> dimension = new ArrayList<Double>();
					String[] clus = val.toString().split(",");
					for (int i = 0; i < clus.length; i++) {
						dimension.add(i, Double.valueOf(clus[i]));
					}
					clusdata.add(dimension);
				}
				ArrayList<Double> centroid = new ArrayList<Double>();
				for (int i = 0; i < clusdata.get(0).size(); i++) {
					Double sum = 0.0;
					for (int j = 0; j < clusdata.size(); j++) {
						sum += clusdata.get(j).get(i);
					}
					sum /= clusdata.size();
					centroid.add(i, sum);
				}
				StringBuilder builder = new StringBuilder();
				for (int i = 0; i < centroid.size(); i++) {
					builder.append(String.valueOf(centroid.get(i)) + " ");
				}
				mos.write("centroid", new Text(builder.toString().substring(0, builder.toString().length() - 1)),
					new Text(" "),"centroid");
				
				// for(Text val: values)
				// 	mos.write("centroid",key,val,"centroid");
			}
		}
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		conf.set("mapred.textoutputformat.separator", " ");

		String outpath = "";
		for(int k=0;k<2;k++)
		{
			for(int j=0;j<2;j++)
			{
				for (int i = 0; i < Integer.parseInt(otherArgs[3]); i++) {
					conf = new Configuration(conf);

					FileSystem fs = FileSystem.get(conf);
					Path pa;
					if (i == 0)
					{
						if(j==0)
							pa = new Path(otherArgs[1]);
						else
							pa = new Path(otherArgs[2]);
					}
					else
						pa = new Path(outpath + "/centroid-r-00000");
					BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(pa)));
					String centroid = "";
					while (br.ready()) {
						centroid += br.readLine() + ",";
					}
					br.close();
					outpath = "/user/root/output/";
					if(k==0)
						outpath += "Eu/";
					else
						outpath += "Ma/";
					if(j==0)
						outpath += "c1/out" + String.valueOf(i + 1);
					else
						outpath += "c2/out" + String.valueOf(i + 1);
					conf.set("centroid", centroid);
					conf.set("dis",String.valueOf(k));
					Job job = Job.getInstance(conf, "Kmeans "+String.valueOf(i));
					job.setJarByClass(Kmeans.class);
					job.setMapperClass(WClusterMapper.class);
			// job.setCombinerClass(IntSumReducer.class);
					job.setMapOutputKeyClass(Text.class);
					job.setMapOutputValueClass(Text.class);
					job.setReducerClass(CentroidReducer.class);
					job.setOutputKeyClass(Text.class);
					job.setOutputValueClass(Text.class);
					MultipleOutputs.addNamedOutput(job, "cost", TextOutputFormat.class, Text.class, Text.class);
					MultipleOutputs.addNamedOutput(job, "centroid", TextOutputFormat.class, Text.class, Text.class);
					FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
					FileOutputFormat.setOutputPath(job, new Path(outpath));
					job.waitForCompletion(true);
				}
			}
		}
		System.exit(0);
	}

}

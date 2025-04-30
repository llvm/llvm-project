#!/usr/bin/python3
# Plot GNU C Library string microbenchmark output.
# Copyright (C) 2019-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
#
# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.
"""Plot string microbenchmark results.

Given a benchmark results file in JSON format and a benchmark schema file,
plot the benchmark timings in one of the available representations.

Separate figure is generated and saved to a file for each 'results' array
found in the benchmark results file. Output filenames and plot titles
are derived from the metadata found in the benchmark results file.
"""
import argparse
from collections import defaultdict
import json
import matplotlib as mpl
import numpy as np
import os

try:
    import jsonschema as validator
except ImportError:
    print("Could not find jsonschema module.")
    raise

# Use pre-selected markers for plotting lines to improve readability
markers = [".", "x", "^", "+", "*", "v", "1", ">", "s"]

# Benchmark variants for which the x-axis scale should be logarithmic
log_variants = {"powers of 2"}


def gmean(numbers):
    """Compute geometric mean.

    Args:
        numbers: 2-D list of numbers
    Return:
        numpy array with geometric means of numbers along each column
    """
    a = np.array(numbers, dtype=np.complex)
    means = a.prod(0) ** (1.0 / len(a))
    return np.real(means)


def relativeDifference(x, x_reference):
    """Compute per-element relative difference between each row of
       a matrix and an array of reference values.

    Args:
        x: numpy matrix of shape (n, m)
        x_reference: numpy array of size m
    Return:
        relative difference between rows of x and x_reference (in %)
    """
    abs_diff = np.subtract(x, x_reference)
    return np.divide(np.multiply(abs_diff, 100.0), x_reference)


def plotTime(timings, routine, bench_variant, title, outpath):
    """Plot absolute timing values.

    Args:
        timings: timings to plot
        routine: benchmarked string routine name
        bench_variant: top-level benchmark variant name
        title: figure title (generated so far)
        outpath: output file path (generated so far)
    Return:
        y: y-axis values to plot
        title_final: final figure title
        outpath_final: file output file path
    """
    y = timings
    plt.figure()

    if not args.values:
        plt.axes().yaxis.set_major_formatter(plt.NullFormatter())

    plt.ylabel("timing")
    title_final = "%s %s benchmark timings\n%s" % \
                  (routine, bench_variant, title)
    outpath_final = os.path.join(args.outdir, "%s_%s_%s%s" % \
                    (routine, args.plot, bench_variant, outpath))

    return y, title_final, outpath_final


def plotRelative(timings, all_timings, routine, ifuncs, bench_variant,
                 title, outpath):
    """Plot timing values relative to a chosen ifunc

    Args:
        timings: timings to plot
        all_timings: all collected timings
        routine: benchmarked string routine name
        ifuncs: names of ifuncs tested
        bench_variant: top-level benchmark variant name
        title: figure title (generated so far)
        outpath: output file path (generated so far)
    Return:
        y: y-axis values to plot
        title_final: final figure title
        outpath_final: file output file path
    """
    # Choose the baseline ifunc
    if args.baseline:
        baseline = args.baseline.replace("__", "")
    else:
        baseline = ifuncs[0]

    baseline_index = ifuncs.index(baseline)

    # Compare timings against the baseline
    y = relativeDifference(timings, all_timings[baseline_index])

    plt.figure()
    plt.axhspan(-args.threshold, args.threshold, color="lightgray", alpha=0.3)
    plt.axhline(0, color="k", linestyle="--", linewidth=0.4)
    plt.ylabel("relative timing (in %)")
    title_final = "Timing comparison against %s\nfor %s benchmark, %s" % \
                  (baseline, bench_variant, title)
    outpath_final = os.path.join(args.outdir, "%s_%s_%s%s" % \
                    (baseline, args.plot, bench_variant, outpath))

    return y, title_final, outpath_final


def plotMax(timings, routine, bench_variant, title, outpath):
    """Plot results as percentage of the maximum ifunc performance.

    The optimal ifunc is computed on a per-parameter-value basis.
    Performance is computed as 1/timing.

    Args:
        timings: timings to plot
        routine: benchmarked string routine name
        bench_variant: top-level benchmark variant name
        title: figure title (generated so far)
        outpath: output file path (generated so far)
    Return:
        y: y-axis values to plot
        title_final: final figure title
        outpath_final: file output file path
    """
    perf = np.reciprocal(timings)
    max_perf = np.max(perf, axis=0)
    y = np.add(100.0, relativeDifference(perf, max_perf))

    plt.figure()
    plt.axhline(100.0, color="k", linestyle="--", linewidth=0.4)
    plt.ylabel("1/timing relative to max (in %)")
    title_final = "Performance comparison against max for %s\n%s " \
                  "benchmark, %s" % (routine, bench_variant, title)
    outpath_final = os.path.join(args.outdir, "%s_%s_%s%s" % \
                    (routine, args.plot, bench_variant, outpath))

    return y, title_final, outpath_final


def plotThroughput(timings, params, routine, bench_variant, title, outpath):
    """Plot throughput.

    Throughput is computed as the varied parameter value over timing.

    Args:
        timings: timings to plot
        params: varied parameter values
        routine: benchmarked string routine name
        bench_variant: top-level benchmark variant name
        title: figure title (generated so far)
        outpath: output file path (generated so far)
    Return:
        y: y-axis values to plot
        title_final: final figure title
        outpath_final: file output file path
    """
    y = np.divide(params, timings)
    plt.figure()

    if not args.values:
        plt.axes().yaxis.set_major_formatter(plt.NullFormatter())

    plt.ylabel("%s / timing" % args.key)
    title_final = "%s %s benchmark throughput results\n%s" % \
                  (routine, bench_variant, title)
    outpath_final = os.path.join(args.outdir, "%s_%s_%s%s" % \
                    (routine, args.plot, bench_variant, outpath))
    return y, title_final, outpath_final


def finishPlot(x, y, title, outpath, x_scale, plotted_ifuncs):
    """Finish generating current Figure.

    Args:
        x: x-axis values
        y: y-axis values
        title: figure title
        outpath: output file path
        x_scale: x-axis scale
        plotted_ifuncs: names of ifuncs to plot
    """
    plt.xlabel(args.key)
    plt.xscale(x_scale)
    plt.title(title)

    plt.grid(color="k", linestyle=args.grid, linewidth=0.5, alpha=0.5)

    for i in range(len(plotted_ifuncs)):
        plt.plot(x, y[i], marker=markers[i % len(markers)],
                 label=plotted_ifuncs[i])

    plt.legend(loc="best", fontsize="small")
    plt.savefig("%s_%s.%s" % (outpath, x_scale, args.extension),
                format=args.extension, dpi=args.resolution)

    if args.display:
        plt.show()

    plt.close()


def plotRecursive(json_iter, routine, ifuncs, bench_variant, title, outpath,
                  x_scale):
    """Plot benchmark timings.

    Args:
        json_iter: reference to json object
        routine: benchmarked string routine name
        ifuncs: names of ifuncs tested
        bench_variant: top-level benchmark variant name
        title: figure's title (generated so far)
        outpath: output file path (generated so far)
        x_scale: x-axis scale
    """

    # RECURSIVE CASE: 'variants' array found
    if "variants" in json_iter:
        # Continue recursive search for 'results' array. Record the
        # benchmark variant (configuration) in order to customize
        # the title, filename and X-axis scale for the generated figure.
        for variant in json_iter["variants"]:
            new_title = "%s%s, " % (title, variant["name"])
            new_outpath = "%s_%s" % (outpath, variant["name"].replace(" ", "_"))
            new_x_scale = "log" if variant["name"] in log_variants else x_scale

            plotRecursive(variant, routine, ifuncs, bench_variant, new_title,
                          new_outpath, new_x_scale)
        return

    # BASE CASE: 'results' array found
    domain = []
    timings = defaultdict(list)

    # Collect timings
    for result in json_iter["results"]:
        domain.append(result[args.key])
        timings[result[args.key]].append(result["timings"])

    domain = np.unique(np.array(domain))
    averages = []

    # Compute geometric mean if there are multple timings for each
    # parameter value.
    for parameter in domain:
        averages.append(gmean(timings[parameter]))

    averages = np.array(averages).transpose()

    # Choose ifuncs to plot
    if isinstance(args.ifuncs, str):
        plotted_ifuncs = ifuncs
    else:
        plotted_ifuncs = [x.replace("__", "") for x in args.ifuncs]

    plotted_indices = [ifuncs.index(x) for x in plotted_ifuncs]
    plotted_vals = averages[plotted_indices,:]

    # Plotting logic specific to each plot type
    if args.plot == "time":
        codomain, title, outpath = plotTime(plotted_vals, routine,
                                   bench_variant, title, outpath)
    elif args.plot == "rel":
        codomain, title, outpath = plotRelative(plotted_vals, averages, routine,
                                   ifuncs, bench_variant, title, outpath)
    elif args.plot == "max":
        codomain, title, outpath = plotMax(plotted_vals, routine,
                                   bench_variant, title, outpath)
    elif args.plot == "thru":
        codomain, title, outpath = plotThroughput(plotted_vals, domain, routine,
                                   bench_variant, title, outpath)

    # Plotting logic shared between plot types
    finishPlot(domain, codomain, title, outpath, x_scale, plotted_ifuncs)


def main(args):
    """Program Entry Point.

    Args:
      args: command line arguments (excluding program name)
    """

    # Select non-GUI matplotlib backend if interactive display is disabled
    if not args.display:
        mpl.use("Agg")

    global plt
    import matplotlib.pyplot as plt

    schema = None

    with open(args.schema, "r") as f:
        schema = json.load(f)

    for filename in args.bench:
        bench = None

        with open(filename, "r") as f:
            bench = json.load(f)

        validator.validate(bench, schema)

        for function in bench["functions"]:
            bench_variant = bench["functions"][function]["bench-variant"]
            ifuncs = bench["functions"][function]["ifuncs"]
            ifuncs = [x.replace("__", "") for x in ifuncs]

            plotRecursive(bench["functions"][function], function, ifuncs,
                          bench_variant, "", "", args.logarithmic)


""" main() """
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=
            "Plot string microbenchmark results",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required parameter
    parser.add_argument("bench", nargs="+",
                        help="benchmark results file(s) in json format")

    # Optional parameters
    parser.add_argument("-b", "--baseline", type=str,
                        help="baseline ifunc for 'rel' plot")
    parser.add_argument("-d", "--display", action="store_true",
                        help="display figures")
    parser.add_argument("-e", "--extension", type=str, default="png",
                        choices=["png", "pdf", "svg"],
                        help="output file(s) extension")
    parser.add_argument("-g", "--grid", action="store_const", default="",
                        const="-", help="show grid lines")
    parser.add_argument("-i", "--ifuncs", nargs="+", default="all",
                        help="ifuncs to plot")
    parser.add_argument("-k", "--key", type=str, default="length",
                        help="key to access the varied parameter")
    parser.add_argument("-l", "--logarithmic", action="store_const",
                        default="linear", const="log",
                        help="use logarithmic x-axis scale")
    parser.add_argument("-o", "--outdir", type=str, default=os.getcwd(),
                        help="output directory")
    parser.add_argument("-p", "--plot", type=str, default="time",
                        choices=["time", "rel", "max", "thru"],
                        help="plot absolute timings, relative timings, " \
                        "performance relative to max, or throughput")
    parser.add_argument("-r", "--resolution", type=int, default=100,
                        help="dpi resolution for the generated figures")
    parser.add_argument("-s", "--schema", type=str,
                        default=os.path.join(os.path.dirname(
                        os.path.realpath(__file__)),
                        "benchout_strings.schema.json"),
                        help="schema file to validate the results file.")
    parser.add_argument("-t", "--threshold", type=int, default=5,
                        help="threshold to mark in 'rel' graph (in %%)")
    parser.add_argument("-v", "--values", action="store_true",
                        help="show actual values")

    args = parser.parse_args()
    main(args)

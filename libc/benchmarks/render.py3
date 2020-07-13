"""Reads JSON files produced by the benchmarking framework and renders them.

Installation:
> apt-get install python3-pip
> pip3 install matplotlib scipy numpy

Run:
> python3 render.py3 <files>

Rendering can occur on disk by specifying the --output option or on screen if
the --headless flag is not set.
"""

import argparse
import collections
import json
import math
import pprint
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
import scipy.stats


def format_freq(number):
    """Returns a human readable frequency."""
    magnitude = 0
    while math.fabs(number) >= 1000:
        number /= 1000.0
        magnitude += 1
    return "%g%sHz" % (number, ["", "k", "M", "G"][magnitude])


def format_size(number):
    """Returns number in human readable form."""
    magnitude = 0
    while number >= 1000 and number % 1000 == 0:
        number /= 1000
        magnitude += 1
    return "%g%s" % (number, ["", "K", "M", "G"][magnitude])


def mean_confidence_interval(dataset, confidence=0.95):
    """Returns the mean and half confidence interval for the dataset."""
    a = 1.0 * np.array(dataset)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def add_plot(function_name, points):
    """Plots measurements for a function."""
    n = len(points.keys())
    x = np.zeros(n)
    y = np.zeros(n)
    yerr = np.zeros(n)

    for i, key in enumerate(sorted(points.keys())):
        values = points[key]
        m, e = mean_confidence_interval(values)
        x[i] = key
        y[i] = m
        yerr[i] = e

    plt.plot(x, y, linewidth=1, label=function_name)
    plt.fill_between(x, y - yerr, y + yerr, alpha=0.5)


def get_title(host):
    """Formats the Host object into a title for the plot."""
    cpu_name = host["CpuName"]
    cpu_freq = format_freq(host["CpuFrequency"])
    cache_strings = []
    for cache in host["Caches"]:
        prefix = {
            "Instruction": "i",
            "Data": "d",
            "Unified": "u",
        }.get(cache["Type"])
        cache_strings.append(r"%sL_%d %s_{/%d}" %
                             (prefix, cache["Level"], format_size(
                                 cache["Size"]), cache["NumSharing"]))
    title = "%s (%s)" % (cpu_name, cpu_freq)
    subtitle = r"$" + ", ".join(sorted(cache_strings)) + r"$"
    return title + "\n" + subtitle


def get_host(jsons):
    """Returns the host of the different json objects iff they are all the same.
    """
    host = None
    for root in jsons:
        if host and host != root["Host"]:
            sys.exit("The datasets are not coming from the same Host")
        if not host:
            host = root["Host"]
    return host


def get_configuration(jsons):
    """Returns the configuration of the different json objects iff they are all
    the same.
    """
    config = None
    for root in jsons:
        if config and config != root["Configuration"]:
            return None
        if not config:
            config = root["Configuration"]
    return config


def setup_graphs(files, display):
    """Setups the graphs to render from the json files."""
    jsons = []
    for file in files:
        with open(file) as json_file:
            jsons.append(json.load(json_file))
    if not jsons:
        sys.exit("Nothing to process")

    for root in jsons:
        frequency = root["Host"]["CpuFrequency"]
        for function in root["Functions"]:
            function_name = function["Name"]
            sizes = function["Sizes"]
            runtimes = function["Runtimes"]
            assert len(sizes) == len(runtimes)
            values = collections.defaultdict(lambda: [])
            for i in range(len(sizes)):
              value = runtimes[i]
              if display == "cycles":
                  value = value * frequency
              if display == "bytespercycle":
                  value = value * frequency
                  value = sizes[i] / value
              values[sizes[i]].append(value)
            add_plot(function_name, values)

    config = get_configuration(jsons)
    if config:
        plt.figtext(
            0.95,
            0.15,
            pprint.pformat(config),
            verticalalignment="bottom",
            horizontalalignment="right",
            multialignment="left",
            fontsize="small",
            bbox=dict(boxstyle="round", facecolor="wheat"))

    axes = plt.gca()
    axes.set_title(get_title(get_host(jsons)))
    axes.set_ylim(bottom=0)
    axes.set_xlabel("Size")
    axes.xaxis.set_major_formatter(EngFormatter(unit="B"))
    if display == "cycles":
          axes.set_ylabel("Cycles")
    if display == "time":
          axes.set_ylabel("Time")
          axes.yaxis.set_major_formatter(EngFormatter(unit="s"))
    if display == "bytespercycle":
          axes.set_ylabel("bytes/cycle")

    plt.legend()
    plt.grid()


def main():
    parser = argparse.ArgumentParser(
        description="Process benchmark json files.")
    parser.add_argument("files", nargs="+", help="The json files to read from.")
    parser.add_argument("--output", help="The output file to write the graph.")
    parser.add_argument(
        "--headless",
        help="If set do not display the graph.",
        action="store_true")
    parser.add_argument(
        "--display",
        choices= ["time", "cycles", "bytespercycle"],
        default="time",
        help="Use to display either 'time', 'cycles' or 'bytes/cycle'.")

    args = parser.parse_args()
    setup_graphs(args.files, args.display)
    if args.output:
        plt.savefig(args.output)
    if not args.headless:
        plt.show()

if __name__ == "__main__":
    main()

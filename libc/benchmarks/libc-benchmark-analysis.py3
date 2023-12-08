"""Reads JSON files produced by the benchmarking framework and renders them.

Installation:
> apt-get install python3-pip
> pip3 install matplotlib pandas seaborn

Run:
> python3 libc/benchmarks/libc-benchmark-analysis.py3 <files>
"""

import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

def formatUnit(value, unit):
    return EngFormatter(unit, sep="").format_data(value)

def formatCache(cache):
  letter = cache["Type"][0].lower()
  level = cache["Level"]
  size = formatUnit(cache["Size"], "B")
  ways = cache["NumSharing"]
  return F'{letter}L{level}:{size}/{ways}'

def getCpuFrequency(study):
    return study["Runtime"]["Host"]["CpuFrequency"]

def getId(study):
    CpuName = study["Runtime"]["Host"]["CpuName"]
    CpuFrequency = formatUnit(getCpuFrequency(study), "Hz")
    Mode = " (Sweep)" if study["Configuration"]["IsSweepMode"] else ""
    CpuCaches = ", ".join(formatCache(c) for c in study["Runtime"]["Host"]["Caches"])
    return F'{CpuName} {CpuFrequency}{Mode}\n{CpuCaches}'

def getFunction(study):
    return study["Configuration"]["Function"]

def getLabel(study):
    return F'{getFunction(study)} {study["StudyName"]}'

def displaySweepData(id, studies, mode):
    df = None
    for study in studies:
        Measurements = study["Measurements"]
        SweepModeMaxSize = study["Configuration"]["SweepModeMaxSize"]
        NumSizes = SweepModeMaxSize + 1
        NumTrials = study["Configuration"]["NumTrials"]
        assert NumTrials * NumSizes  == len(Measurements), 'not a multiple of NumSizes'
        Index = pd.MultiIndex.from_product([range(NumSizes), range(NumTrials)], names=['size', 'trial'])
        if df is None:
            df = pd.DataFrame(Measurements, index=Index, columns=[getLabel(study)])
        else:
            df[getLabel(study)] = pd.Series(Measurements, index=Index)
    df = df.reset_index(level='trial', drop=True)
    if mode == "cycles":
        df *= getCpuFrequency(study)
    if mode == "bytespercycle":
        df *= getCpuFrequency(study)
        for col in df.columns:
            df[col] = pd.Series(data=df.index, index=df.index).divide(df[col])
    FormatterUnit = {"time":"s","cycles":"","bytespercycle":"B/cycle"}[mode]
    Label = {"time":"Time","cycles":"Cycles","bytespercycle":"Byte/cycle"}[mode]
    graph = sns.lineplot(data=df, palette="muted", ci=95)
    graph.set_title(id)
    graph.yaxis.set_major_formatter(EngFormatter(unit=FormatterUnit))
    graph.yaxis.set_label_text(Label)
    graph.xaxis.set_major_formatter(EngFormatter(unit="B"))
    graph.xaxis.set_label_text("Copy Size")
    _ = plt.xticks(rotation=90)
    plt.show()

def displayDistributionData(id, studies, mode):
    distributions = set()
    df = None
    for study in studies:
        distribution = study["Configuration"]["SizeDistributionName"]
        distributions.add(distribution)
        local = pd.DataFrame(study["Measurements"], columns=["time"])
        local["distribution"] = distribution
        local["label"] = getLabel(study)
        local["cycles"] = local["time"] * getCpuFrequency(study)
        if df is None:
            df = local
        else:
            df = df.append(local)
    if mode == "bytespercycle":
        mode = "time"
        print("`--mode=bytespercycle` is ignored for distribution mode reports")
    FormatterUnit = {"time":"s","cycles":""}[mode]
    Label = {"time":"Time","cycles":"Cycles"}[mode]
    graph = sns.violinplot(data=df, x="distribution", y=mode, palette="muted", hue="label", order=sorted(distributions))
    graph.set_title(id)
    graph.yaxis.set_major_formatter(EngFormatter(unit=FormatterUnit))
    graph.yaxis.set_label_text(Label)
    _ = plt.xticks(rotation=90)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Process benchmark json files.")
    parser.add_argument("--mode", choices=["time", "cycles", "bytespercycle"], default="time", help="Use to display either 'time', 'cycles' or 'bytes/cycle'.")
    parser.add_argument("files", nargs="+", help="The json files to read from.")

    args = parser.parse_args()
    study_groups = dict()
    for file in args.files:
        with open(file) as json_file:
            json_obj = json.load(json_file)
            Id = getId(json_obj)
            if Id in study_groups:
                study_groups[Id].append(json_obj)
            else:
                study_groups[Id] = [json_obj]

    plt.tight_layout()
    sns.set_theme(style="ticks")
    for id, study_collection in study_groups.items():
        if "(Sweep)" in id:
            displaySweepData(id, study_collection, args.mode)
        else:
            displayDistributionData(id, study_collection, args.mode)


if __name__ == "__main__":
    main()

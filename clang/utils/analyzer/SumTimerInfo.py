#!/usr/bin/env python

"""
Script to Summarize statistics in the scan-build output.

Statistics are enabled by passing '-internal-stats' option to scan-build
(or '-analyzer-stats' to the analyzer).
"""

import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print >> sys.stderr, 'Usage: ', sys.argv[0],\
                             'scan_build_output_file'
        sys.exit(-1)

    f = open(sys.argv[1], 'r')
    Time = 0.0
    TotalTime = 0.0
    MaxTime = 0.0
    Warnings = 0
    Count = 0
    FunctionsAnalyzed = 0
    ReachableBlocks = 0
    ReachedMaxSteps = 0
    NumSteps = 0
    NumInlinedCallSites = 0
    NumBifurcatedCallSites = 0
    MaxCFGSize = 0
    for line in f:
        if ("Analyzer Total Time" in line):
            s = line.split()
            Time = Time + float(s[6])
            Count = Count + 1
            if (float(s[6]) > MaxTime):
                MaxTime = float(s[6])
        if ("warning generated." in line) or ("warnings generated" in line):
            s = line.split()
            Warnings = Warnings + int(s[0])
        if "The # of functions analysed (as top level)" in line:
            s = line.split()
            FunctionsAnalyzed = FunctionsAnalyzed + int(s[0])
        if "The % of reachable basic blocks" in line:
            s = line.split()
            ReachableBlocks = ReachableBlocks + int(s[0])
        if "The # of times we reached the max number of steps" in line:
            s = line.split()
            ReachedMaxSteps = ReachedMaxSteps + int(s[0])
        if "The maximum number of basic blocks in a function" in line:
            s = line.split()
            if MaxCFGSize < int(s[0]):
                MaxCFGSize = int(s[0])
        if "The # of steps executed" in line:
            s = line.split()
            NumSteps = NumSteps + int(s[0])
        if "The # of times we inlined a call" in line:
            s = line.split()
            NumInlinedCallSites = NumInlinedCallSites + int(s[0])
        if "The # of times we split the path due \
                to imprecise dynamic dispatch info" in line:
            s = line.split()
            NumBifurcatedCallSites = NumBifurcatedCallSites + int(s[0])
        if ")  Total" in line:
            s = line.split()
            TotalTime = TotalTime + float(s[6])

    print "TU Count %d" % (Count)
    print "Time %f" % (Time)
    print "Warnings %d" % (Warnings)
    print "Functions Analyzed %d" % (FunctionsAnalyzed)
    print "Reachable Blocks %d" % (ReachableBlocks)
    print "Reached Max Steps %d" % (ReachedMaxSteps)
    print "Number of Steps %d" % (NumSteps)
    print "Number of Inlined calls %d (bifurcated %d)" % (
        NumInlinedCallSites, NumBifurcatedCallSites)
    print "MaxTime %f" % (MaxTime)
    print "TotalTime %f" % (TotalTime)
    print "Max CFG Size %d" % (MaxCFGSize)

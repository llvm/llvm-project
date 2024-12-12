//===- bolt/Utils/CommandLineOpts.cpp - BOLT CLI options ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BOLT CLI options
//
//===----------------------------------------------------------------------===//

#include "bolt/Utils/CommandLineOpts.h"
#include "VCSVersion.inc"

using namespace llvm;

namespace llvm {
namespace bolt {
const char *BoltRevision =
#ifdef BOLT_REVISION
    BOLT_REVISION;
#else
    "<unknown>";
#endif
}
}

namespace opts {

bool HeatmapMode = false;
bool BinaryAnalysisMode = false;

cl::OptionCategory BoltCategory("BOLT generic options");
cl::OptionCategory BoltDiffCategory("BOLTDIFF generic options");
cl::OptionCategory BoltOptCategory("BOLT optimization options");
cl::OptionCategory BoltRelocCategory("BOLT options in relocation mode");
cl::OptionCategory BoltOutputCategory("Output options");
cl::OptionCategory AggregatorCategory("Data aggregation options");
cl::OptionCategory BoltInstrCategory("BOLT instrumentation options");
cl::OptionCategory HeatmapCategory("Heatmap options");
cl::OptionCategory BinaryAnalysisCategory("BinaryAnalysis options");

cl::opt<unsigned> AlignText("align-text",
                            cl::desc("alignment of .text section"), cl::Hidden,
                            cl::cat(BoltCategory));

cl::opt<unsigned> AlignFunctions(
    "align-functions",
    cl::desc("align functions at a given value (relocation mode)"),
    cl::init(64), cl::cat(BoltOptCategory));

cl::opt<bool>
AggregateOnly("aggregate-only",
  cl::desc("exit after writing aggregated data file"),
  cl::Hidden,
  cl::cat(AggregatorCategory));

cl::opt<unsigned>
    BucketsPerLine("line-size",
                   cl::desc("number of entries per line (default 256)"),
                   cl::init(256), cl::Optional, cl::cat(HeatmapCategory));

cl::opt<bool>
DiffOnly("diff-only",
  cl::desc("stop processing once we have enough to compare two binaries"),
  cl::Hidden,
  cl::cat(BoltDiffCategory));

cl::opt<bool>
EnableBAT("enable-bat",
  cl::desc("write BOLT Address Translation tables"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(BoltCategory));

cl::opt<bool> EqualizeBBCounts(
    "equalize-bb-counts",
    cl::desc("use same count for BBs that should have equivalent count (used "
             "in non-LBR and shrink wrapping)"),
    cl::ZeroOrMore, cl::init(false), cl::Hidden, cl::cat(BoltOptCategory));

cl::opt<bool> RemoveSymtab("remove-symtab", cl::desc("Remove .symtab section"),
                           cl::cat(BoltCategory));

cl::opt<unsigned>
ExecutionCountThreshold("execution-count-threshold",
  cl::desc("perform profiling accuracy-sensitive optimizations only if "
           "function execution count >= the threshold (default: 0)"),
  cl::init(0),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
    HeatmapBlock("block-size",
                 cl::desc("size of a heat map block in bytes (default 64)"),
                 cl::init(64), cl::cat(HeatmapCategory));

cl::opt<unsigned long long> HeatmapMaxAddress(
    "max-address", cl::init(0xffffffff),
    cl::desc("maximum address considered valid for heatmap (default 4GB)"),
    cl::Optional, cl::cat(HeatmapCategory));

cl::opt<unsigned long long> HeatmapMinAddress(
    "min-address", cl::init(0x0),
    cl::desc("minimum address considered valid for heatmap (default 0)"),
    cl::Optional, cl::cat(HeatmapCategory));

cl::opt<bool> HeatmapPrintMappings(
    "print-mappings", cl::init(false),
    cl::desc("print mappings in the legend, between characters/blocks and text "
             "sections (default false)"),
    cl::Optional, cl::cat(HeatmapCategory));

cl::opt<bool> HotData("hot-data",
                      cl::desc("hot data symbols support (relocation mode)"),
                      cl::cat(BoltCategory));

cl::opt<bool> HotFunctionsAtEnd(
    "hot-functions-at-end",
    cl::desc(
        "if reorder-functions is used, order functions putting hottest last"),
    cl::cat(BoltCategory));

cl::opt<bool> HotText(
    "hot-text",
    cl::desc(
        "Generate hot text symbols. Apply this option to a precompiled binary "
        "that manually calls into hugify, such that at runtime hugify call "
        "will put hot code into 2M pages. This requires relocation."),
    cl::ZeroOrMore, cl::cat(BoltCategory));

cl::opt<bool>
    Instrument("instrument",
               cl::desc("instrument code to generate accurate profile data"),
               cl::cat(BoltOptCategory));

cl::opt<bool> Lite("lite", cl::desc("skip processing of cold functions"),
                   cl::cat(BoltCategory));

cl::opt<std::string>
OutputFilename("o",
  cl::desc("<output file>"),
  cl::Optional,
  cl::cat(BoltOutputCategory));

cl::opt<std::string> PerfData("perfdata", cl::desc("<data file>"), cl::Optional,
                              cl::cat(AggregatorCategory),
                              cl::sub(cl::SubCommand::getAll()));

static cl::alias
PerfDataA("p",
  cl::desc("alias for -perfdata"),
  cl::aliasopt(PerfData),
  cl::cat(AggregatorCategory));

cl::opt<bool> PrintCacheMetrics(
    "print-cache-metrics",
    cl::desc("calculate and print various metrics for instruction cache"),
    cl::cat(BoltOptCategory));

cl::opt<bool> PrintSections("print-sections",
                            cl::desc("print all registered sections"),
                            cl::Hidden, cl::cat(BoltCategory));

cl::opt<ProfileFormatKind> ProfileFormat(
    "profile-format",
    cl::desc(
        "format to dump profile output in aggregation mode, default is fdata"),
    cl::init(PF_Fdata),
    cl::values(clEnumValN(PF_Fdata, "fdata", "offset-based plaintext format"),
               clEnumValN(PF_YAML, "yaml", "dense YAML representation")),
    cl::ZeroOrMore, cl::Hidden, cl::cat(BoltCategory));

cl::opt<std::string> SaveProfile("w",
                                 cl::desc("save recorded profile to a file"),
                                 cl::cat(BoltOutputCategory));

cl::opt<bool> ShowDensity("show-density",
                          cl::desc("show profile density details"),
                          cl::Optional, cl::cat(AggregatorCategory));

cl::opt<bool> SplitEH("split-eh", cl::desc("split C++ exception handling code"),
                      cl::Hidden, cl::cat(BoltOptCategory));

cl::opt<bool>
    StrictMode("strict",
               cl::desc("trust the input to be from a well-formed source"),

               cl::cat(BoltCategory));

cl::opt<bool> TimeOpts("time-opts",
                       cl::desc("print time spent in each optimization"),
                       cl::cat(BoltOptCategory));

cl::opt<bool> TimeRewrite("time-rewrite",
                          cl::desc("print time spent in rewriting passes"),
                          cl::Hidden, cl::cat(BoltCategory));

cl::opt<bool> UseOldText(
    "use-old-text",
    cl::desc("re-use space in old .text if possible (relocation mode)"),
    cl::cat(BoltCategory));

cl::opt<bool> UpdateDebugSections(
    "update-debug-sections",
    cl::desc("update DWARF debug sections of the executable"),
    cl::cat(BoltCategory));

cl::opt<unsigned>
    Verbosity("v", cl::desc("set verbosity level for diagnostic output"),
              cl::init(0), cl::ZeroOrMore, cl::cat(BoltCategory),
              cl::sub(cl::SubCommand::getAll()));

bool processAllFunctions() {
  if (opts::AggregateOnly)
    return false;

  if (UseOldText || StrictMode)
    return true;

  return false;
}

} // namespace opts

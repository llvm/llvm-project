//===- bolt/Utils/CommandLineOpts.h - BOLT CLI options ----------*- C++ -*-===//
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

#ifndef BOLT_UTILS_COMMAND_LINE_OPTS_H
#define BOLT_UTILS_COMMAND_LINE_OPTS_H

#include "llvm/Support/CommandLine.h"

namespace llvm {
namespace bolt {
class BinaryFunction;
}
} // namespace llvm

namespace opts {

enum HeatmapModeKind {
  HM_None = 0,
  HM_Exclusive, // llvm-bolt-heatmap
  HM_Optional   // perf2bolt --heatmap
};

using HeatmapBlockSizes = std::vector<unsigned>;
struct HeatmapBlockSpecParser : public llvm::cl::parser<HeatmapBlockSizes> {
  explicit HeatmapBlockSpecParser(llvm::cl::Option &O)
      : llvm::cl::parser<HeatmapBlockSizes>(O) {}
  // Return true on error.
  bool parse(llvm::cl::Option &O, llvm::StringRef ArgName, llvm::StringRef Arg,
             HeatmapBlockSizes &Val);
};

extern HeatmapModeKind HeatmapMode;
extern bool BinaryAnalysisMode;

extern llvm::cl::OptionCategory BoltCategory;
extern llvm::cl::OptionCategory BoltDiffCategory;
extern llvm::cl::OptionCategory BoltOptCategory;
extern llvm::cl::OptionCategory BoltRelocCategory;
extern llvm::cl::OptionCategory BoltOutputCategory;
extern llvm::cl::OptionCategory AggregatorCategory;
extern llvm::cl::OptionCategory BoltInstrCategory;
extern llvm::cl::OptionCategory HeatmapCategory;
extern llvm::cl::OptionCategory BinaryAnalysisCategory;

extern llvm::cl::opt<unsigned> AlignText;
extern llvm::cl::opt<unsigned> AlignFunctions;
extern llvm::cl::opt<bool> AggregateOnly;
extern llvm::cl::opt<bool> ArmSPE;
extern llvm::cl::opt<unsigned> BucketsPerLine;
extern llvm::cl::opt<bool> CompactCodeModel;
extern llvm::cl::opt<bool> DiffOnly;
extern llvm::cl::opt<bool> EnableBAT;
extern llvm::cl::opt<bool> EqualizeBBCounts;
extern llvm::cl::opt<bool> ForcePatch;
extern llvm::cl::opt<bool> RemoveSymtab;
extern llvm::cl::opt<unsigned> ExecutionCountThreshold;
extern llvm::cl::opt<HeatmapBlockSizes, false, HeatmapBlockSpecParser>
    HeatmapBlock;
extern llvm::cl::opt<unsigned long long> HeatmapMaxAddress;
extern llvm::cl::opt<unsigned long long> HeatmapMinAddress;
extern llvm::cl::opt<bool> HeatmapPrintMappings;
extern llvm::cl::opt<std::string> HeatmapOutput;
extern llvm::cl::opt<bool> HotData;
extern llvm::cl::opt<bool> HotFunctionsAtEnd;
extern llvm::cl::opt<bool> HotText;
extern llvm::cl::opt<bool> Hugify;
extern llvm::cl::opt<bool> Instrument;
extern llvm::cl::opt<std::string> OutputFilename;
extern llvm::cl::opt<std::string> PerfData;
extern llvm::cl::opt<bool> PrintCacheMetrics;
extern llvm::cl::opt<bool> PrintSections;

// The format to use with -o in aggregation mode (perf2bolt)
enum ProfileFormatKind { PF_Fdata, PF_YAML };

extern llvm::cl::opt<ProfileFormatKind> ProfileFormat;
extern llvm::cl::opt<bool> ShowDensity;
extern llvm::cl::opt<bool> SplitEH;
extern llvm::cl::opt<bool> StrictMode;
extern llvm::cl::opt<bool> TimeOpts;
extern llvm::cl::opt<bool> UseOldText;
extern llvm::cl::opt<bool> UpdateDebugSections;

// The default verbosity level (0) is pretty terse, level 1 is fairly
// verbose and usually prints some informational message for every
// function processed.  Level 2 is for the noisiest of messages and
// often prints a message per basic block.
// Error messages should never be suppressed by the verbosity level.
// Only warnings and info messages should be affected.
//
// The rationale behind stream usage is as follows:
// outs() for info and debugging controlled by command line flags.
// errs() for errors and warnings.
// dbgs() for output within DEBUG().
extern llvm::cl::opt<unsigned> Verbosity;

/// Return true if we should process all functions in the binary.
bool processAllFunctions();

/// Return true if we should dump dot graphs for the given function.
bool shouldDumpDot(const llvm::bolt::BinaryFunction &Function);

enum GadgetScannerKind { GS_PACRET, GS_PAUTH, GS_ALL };

extern llvm::cl::bits<GadgetScannerKind> GadgetScannersToRun;

} // namespace opts

namespace llvm {
namespace bolt {
extern const char *BoltRevision;
}
} // namespace llvm

#endif

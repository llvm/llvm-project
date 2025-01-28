//===- bolt/Passes/ContinuityStats.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass checks how well the BOLT input profile satisfies the following
// "CFG continuity" property of a perfect profile:
//
//        Each positive-execution-count block in the function’s CFG
//        should be *reachable* from a positive-execution-count function
//        entry block through a positive-execution-count path.
//
// More specifically, for each of the hottest 1000 functions, the pass
// calculates the function’s fraction of basic block execution counts
// that is *unreachable*. It then reports the 95th percentile of the
// distribution of the 1000 unreachable fractions in a single BOLT-INFO line.
// The smaller the reported value is, the better the BOLT profile
// satisfies the CFG continuity property.

// The default value of 1000 above can be changed via the hidden BOLT option
// `-num-functions-for-continuity-check=[N]`.
// If more detailed stats are needed, `-v=1` can be used: the hottest N
// functions will be grouped into 5 equally-sized buckets, from the hottest
// to the coldest; for each bucket, various summary statistics of the
// distribution of the unreachable fractions and the raw unreachable execution
// counts will be reported.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_CONTINUITYSTATS_H
#define BOLT_PASSES_CONTINUITYSTATS_H

#include "bolt/Passes/BinaryPasses.h"
#include <vector>

namespace llvm {

class raw_ostream;

namespace bolt {
class BinaryContext;

/// Compute and report to the user the function CFG continuity quality
class PrintContinuityStats : public BinaryFunctionPass {
public:
  explicit PrintContinuityStats(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  bool shouldOptimize(const BinaryFunction &BF) const override;
  const char *getName() const override { return "continuity-stats"; }
  bool shouldPrint(const BinaryFunction &) const override { return false; }
  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif // BOLT_PASSES_CONTINUITYSTATS_H

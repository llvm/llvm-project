//===- bolt/Passes/ProfileQualityStats.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass checks the BOLT input profile quality.
//
// Check 1: how well the input profile satisfies the following
// "CFG continuity" property of a perfect profile:
//
//        Each positive-execution-count block in the function’s CFG
//        is *reachable* from a positive-execution-count function
//        entry block through a positive-execution-count path.
//
// More specifically, for each of the hottest 1000 functions, the pass
// calculates the function’s fraction of basic block execution counts
// that is *unreachable*. It then reports the 95th percentile of the
// distribution of the 1000 unreachable fractions in a single BOLT-INFO line.
// The smaller the reported value is, the better the BOLT profile
// satisfies the CFG continuity property.
//
// Check 2: how well the input profile satisfies the "call graph flow
// conservation" property of a perfect profile:
//
//        For each function that is not a program entry, the number of times the
//        function is called is equal to the net CFG outflow of the
//        function's entry block(s).
//
// More specifically, for each of the hottest 1000 functions, the pass obtains
// A = number of times the function is called, B = the function's entry blocks'
// inflow, C = the function's entry blocks' outflow, where B and C are computed
// using the function's weighted CFG. It then computes gap = 1 - MIN(A,C-B) /
// MAX(A, C-B). The pass reports the 95th percentile of the distribution of the
// 1000 gaps in a single BOLT-INFO line. The smaller the reported value is, the
// better the BOLT profile satisfies the call graph flow conservation property.
//
// Check 3: how well the input profile satisfies the "function CFG flow
// conservation property" of a perfect profile:
//
//       A non-entry non-exit basic block's inflow is equal to its outflow.
//
// More specifically, for each of the hottest 1000 functions, the pass loops
// over its basic blocks that are non-entry and non-exit, and for each block
// obtains a block gap = 1 - MIN(block inflow, block outflow, block call count
// if any) / MAX(block inflow, block outflow, block call count if any). It then
// aggregates the block gaps into 2 values for the function: "weighted" is the
// weighted average of the block conservation gaps, where the weights depend on
// each block's execution count and instruction count; "worst" is the worst
// (biggest) block gap across all basic blocks in the function with an execution
// count of > 500. The pass then reports the 95th percentile of the weighted and
// worst values of the 1000 functions in a single BOLT-INFO line. The smaller
// the reported values are, the better the BOLT profile satisfies the function
// CFG flow conservation property.
//
// The default value of 1000 above can be changed via the hidden BOLT option
// `-top-functions-for-profile-quality-check=[N]`.
// The default reporting of the 95th percentile can be changed via the hidden
// BOLT option `-percentile-for-profile-quality-check=[M]`.
//
// If more detailed stats are needed, `-v=1` can be used: the hottest N
// functions will be grouped into 5 equally-sized buckets, from the hottest
// to the coldest; for each bucket, various summary statistics of the
// profile quality will be reported.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_PROFILEQUALITYSTATS_H
#define BOLT_PASSES_PROFILEQUALITYSTATS_H

#include "bolt/Passes/BinaryPasses.h"
#include <vector>

namespace llvm {

class raw_ostream;

namespace bolt {
class BinaryContext;

/// Compute and report to the user the profile quality
class PrintProfileQualityStats : public BinaryFunctionPass {
public:
  explicit PrintProfileQualityStats(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  bool shouldOptimize(const BinaryFunction &BF) const override;
  const char *getName() const override { return "profile-quality-stats"; }
  bool shouldPrint(const BinaryFunction &) const override { return false; }
  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif // BOLT_PASSES_PROFILEQUALITYSTATS_H

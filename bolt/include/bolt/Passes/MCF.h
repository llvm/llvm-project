//===- bolt/Passes/MCF.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_MCF_H
#define BOLT_PASSES_MCF_H

namespace llvm {
namespace bolt {

class BinaryFunction;
class DataflowInfoManager;

enum MCFCostFunction : char {
  MCF_DISABLE = 0,
  MCF_LINEAR,
  MCF_QUADRATIC,
  MCF_LOG,
  MCF_BLAMEFTS
};

/// Implement the idea in "SamplePGO - The Power of Profile Guided Optimizations
/// without the Usability Burden" by Diego Novillo to make basic block counts
/// equal if we show that A dominates B, B post-dominates A and they are in the
/// same loop and same loop nesting level.
void equalizeBBCounts(DataflowInfoManager &Info, BinaryFunction &BF);

/// Fill edge counts based on the basic block count. Used in nonLBR mode when
/// we only have bb count.
void estimateEdgeCounts(BinaryFunction &BF);

/// Entry point for computing a min-cost flow for the CFG with the goal
/// of fixing the flow of the CFG edges, that is, making sure it obeys the
/// flow-conservation equation  SumInEdges = SumOutEdges.
///
/// To do this, we create an instance of the min-cost flow problem in a
/// similar way as the one discussed in the work of Roy Levin "Completing
/// Incomplete Edge Profile by Applying Minimum Cost Circulation Algorithms".
/// We do a few things differently, though. We don't populate edge counts using
/// weights coming from a static branch prediction technique and we don't
/// use the same cost function.
///
/// If cost function BlameFTs is used, assign all remaining flow to
/// fall-throughs. This is used when the sampling is based on taken branches
/// that do not account for them.
void solveMCF(BinaryFunction &BF, MCFCostFunction CostFunction);

} // end namespace bolt
} // end namespace llvm

#endif // BOLT_PASSES_MCF_H

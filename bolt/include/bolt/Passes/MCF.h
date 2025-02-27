//===- bolt/Passes/MCF.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_MCF_H
#define BOLT_PASSES_MCF_H

#include "bolt/Passes/BinaryPasses.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {
namespace bolt {

class DataflowInfoManager;

/// Implement the idea in "SamplePGO - The Power of Profile Guided Optimizations
/// without the Usability Burden" by Diego Novillo to make basic block counts
/// equal if we show that A dominates B, B post-dominates A and they are in the
/// same loop and same loop nesting level.
void equalizeBBCounts(DataflowInfoManager &Info, BinaryFunction &BF);

/// Fill edge counts based on the basic block count. Used in nonLBR mode when
/// we only have bb count.
class EstimateEdgeCounts : public BinaryFunctionPass {
  void runOnFunction(BinaryFunction &BF);

public:
  explicit EstimateEdgeCounts(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "estimate-edge-counts"; }

  /// Pass entry point
  Error runOnFunctions(BinaryContext &BC) override;
};

} // end namespace bolt
} // end namespace llvm

#endif // BOLT_PASSES_MCF_H

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

/// Implement the idea in "SamplePGO - The Power of Profile Guided Optimizations
/// without the Usability Burden" by Diego Novillo to make basic block counts
/// equal if we show that A dominates B, B post-dominates A and they are in the
/// same loop and same loop nesting level.
void equalizeBBCounts(DataflowInfoManager &Info, BinaryFunction &BF);

/// Fill edge counts based on the basic block count. Used in nonLBR mode when
/// we only have bb count.
void estimateEdgeCounts(BinaryFunction &BF);

} // end namespace bolt
} // end namespace llvm

#endif // BOLT_PASSES_MCF_H

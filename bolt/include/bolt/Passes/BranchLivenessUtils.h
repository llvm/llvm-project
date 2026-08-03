//===- bolt/Passes/BranchLivenessUtils.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_BRANCHLIVENESSUTILS_H
#define BOLT_PASSES_BRANCHLIVENESSUTILS_H

#include "llvm/ADT/DenseSet.h"

namespace llvm {
class MCInst;

namespace bolt {
class BinaryFunction;
class RegAnalysis;

/// Return true if \p BF has short-range branches.
bool hasShortRangeBranch(BinaryFunction &BF);

/// Return the branch instructions where the target flags register is dead.
DenseSet<const MCInst *> computeDeadFlagBranches(BinaryFunction &BF,
                                                 RegAnalysis &RA);

} // namespace bolt
} // namespace llvm

#endif

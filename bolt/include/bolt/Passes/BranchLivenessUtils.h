//===- bolt/Passes/BranchLivenessUtils.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_BRANCHLIVENESSUTILS_H
#define BOLT_PASSES_BRANCHLIVENESSUTILS_H

#include "bolt/Core/BranchLivenessInfo.h"

namespace llvm {
namespace bolt {
class BinaryFunction;
class RegAnalysis;

/// Return true if \p BF needs liveness info for branch transformations.
bool needsBranchLiveness(BinaryFunction &BF);

/// Return liveness info required for branch transformations.
BranchLivenessInfo computeBranchLiveness(BinaryFunction &BF, RegAnalysis &RA);

} // namespace bolt
} // namespace llvm

#endif

//===- bolt/Core/BranchLiveness.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_BRANCHLIVENESS_H
#define BOLT_CORE_BRANCHLIVENESS_H

#include "llvm/ADT/DenseSet.h"

namespace llvm {
class MCInst;

namespace bolt {

struct BranchLivenessInfo {
  DenseSet<const MCInst *> BranchesWithDeadFlags;

  bool mustPreserveFlags(const MCInst &Inst) const {
    return !BranchesWithDeadFlags.count(&Inst);
  }
};

} // namespace bolt
} // namespace llvm

#endif

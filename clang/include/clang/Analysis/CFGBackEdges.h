//===- CFGBackEdges.h - Finds back edges in Clang CFGs -*- C++ ----------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_CFG_BACKEDGES_H
#define LLVM_CLANG_ANALYSIS_CFG_BACKEDGES_H

#include "clang/Analysis/CFG.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {

/// Finds and returns back edges in Clang CFGs. The CFG already has some
/// backedge information for structured loops (\c CFGBlock::getLoopTarget).
/// However, unstructured back edges from \c goto statements are not included.
/// This helps find back edges, whether the CFG is reducible or not.
/// This includes CFGBlock::getLoopTarget nodes, but one can filter those out.
llvm::DenseMap<const CFGBlock *, const CFGBlock *>
findCFGBackEdges(const CFG &CFG);

} // namespace clang

#endif  // LLVM_CLANG_ANALYSIS_CFG_BACKEDGES_H

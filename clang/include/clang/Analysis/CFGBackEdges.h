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
#include "llvm/ADT/DenseSet.h"

namespace clang {

/// Finds and returns back edges in Clang CFGs. The CFG already has some
/// backedge information for structured loops (\c CFGBlock::getLoopTarget).
/// However, unstructured back edges from \c goto statements are not included.
/// This helps find back edges, whether the CFG is reducible or not.
/// This includes CFGBlock::getLoopTarget nodes, but one can filter those out
/// e.g., with \c findNonStructuredLoopBackedgeNodes.
llvm::DenseMap<const CFGBlock *, const CFGBlock *>
findCFGBackEdges(const CFG &CFG);

/// Returns a set of CFG blocks that is the source of a backedge and is not
/// tracked as part of a structured loop (with `CFGBlock::getLoopTarget`).
llvm::SmallDenseSet<const CFGBlock *>
findNonStructuredLoopBackedgeNodes(const CFG &CFG);

/// Given a backedge from B1 to B2, B1 is a "backedge node" in a CFG.
/// It can be:
/// - A block introduced in the CFG exclusively to indicate a structured loop's
///   backedge. They are exactly identified by the presence of a non-null
///   pointer to the entry block of the loop condition. Note that this is not
///   necessarily the block with the loop statement as terminator, because
///   short-circuit operators will result in multiple blocks encoding the loop
///   condition, only one of which will contain the loop statement as
///   terminator.
/// - A block that is part of a backedge in a CFG with unstructured loops
///   (e.g., a CFG with a `goto` statement). Note that this is not necessarily
///   the block with the goto statement as terminator. The choice depends on how
///   blocks and edges are ordered.
///
/// \param NonStructLoopBackedgeNodes is the set of nodes from
/// \c findNonStructuredLoopBackedgeNodes.
bool isBackedgeCFGNode(
    const CFGBlock &B,
    const llvm::SmallDenseSet<const CFGBlock *> &NonStructLoopBackedgeNodes);

} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_CFG_BACKEDGES_H

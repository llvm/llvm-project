//===- CFGBackEdges.cpp - Finds back edges in Clang CFGs ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stack>
#include <utility>
#include <vector>

#include "clang/Analysis/CFG.h"
#include "clang/Analysis/CFGBackEdges.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {

namespace {
struct VisitClockTimes {
  // Timestamp for when the node was visited / discovered.
  int Pre = -1;
  // Timestamp for when we finished visiting a node's successors.
  int Post = -1;
};
} // namespace

// Returns true if the CFG contains any goto statements (direct or indirect).
static bool hasGotoInCFG(const CFG &CFG) {
  for (const CFGBlock *Block : CFG) {
    const Stmt *Term = Block->getTerminatorStmt();
    if (Term == nullptr)
      continue;
    if (isa<GotoStmt>(Term) || isa<IndirectGotoStmt>(Term))
      return true;
  }
  return false;
}

llvm::DenseMap<const CFGBlock *, const CFGBlock *>
findCFGBackEdges(const CFG &CFG) {
  // Do a simple textbook DFS with pre and post numberings to find back edges.
  llvm::DenseMap<const CFGBlock *, const CFGBlock *> BackEdges;

  std::vector<VisitClockTimes> VisitState;
  VisitState.resize(CFG.getNumBlockIDs());
  std::stack<std::pair<const CFGBlock *, CFGBlock::const_succ_iterator>>
      DFSStack;
  int Clock = 0;
  const CFGBlock &Entry = CFG.getEntry();
  VisitState[Entry.getBlockID()].Pre = Clock++;
  DFSStack.push({&Entry, Entry.succ_begin()});

  while (!DFSStack.empty()) {
    auto &[Block, SuccIt] = DFSStack.top();
    if (SuccIt == Block->succ_end()) {
      VisitState[Block->getBlockID()].Post = Clock++;
      DFSStack.pop();
      continue;
    }

    const CFGBlock::AdjacentBlock &AdjacentSucc = *SuccIt++;
    const CFGBlock *Succ = AdjacentSucc.getReachableBlock();
    // Skip unreachable blocks.
    if (Succ == nullptr)
      continue;

    VisitClockTimes &SuccVisitState = VisitState[Succ->getBlockID()];
    if (SuccVisitState.Pre != -1) {
      if (SuccVisitState.Post == -1)
        BackEdges.insert({Block, Succ});
    } else {
      SuccVisitState.Pre = Clock++;
      DFSStack.push({Succ, Succ->succ_begin()});
    }
  }
  return BackEdges;
}

// Returns a set of CFG blocks that is the source of a backedge and is not
// tracked as part of a structured loop (with `CFGBlock::getLoopTarget`).
llvm::SmallDenseSet<const CFGBlock *>
findNonStructuredLoopBackedgeNodes(const CFG &CFG) {
  llvm::SmallDenseSet<const CFGBlock *> NonStructLoopBackedgeNodes;
  // We should only need this if the function has gotos.
  if (!hasGotoInCFG(CFG))
    return NonStructLoopBackedgeNodes;

  llvm::DenseMap<const CFGBlock *, const CFGBlock *> Backedges =
      findCFGBackEdges(CFG);
  for (const auto &[From, To] : Backedges) {
    if (From->getLoopTarget() == nullptr)
      NonStructLoopBackedgeNodes.insert(From);
  }
  return NonStructLoopBackedgeNodes;
}

bool isBackedgeCFGNode(
    const CFGBlock &B,
    const llvm::SmallDenseSet<const CFGBlock *> &NonStructLoopBackedgeNodes) {
  return B.getLoopTarget() != nullptr ||
         NonStructLoopBackedgeNodes.contains(&B);
}

} // namespace clang

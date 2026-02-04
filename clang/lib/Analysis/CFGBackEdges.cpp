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

llvm::DenseMap<const CFGBlock *, const CFGBlock *>
findCFGBackEdges(const clang::CFG &CFG) {
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

} // namespace clang

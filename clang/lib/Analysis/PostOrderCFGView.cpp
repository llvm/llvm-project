//===- PostOrderCFGView.cpp - Post order view of CFG blocks ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements post order view of the blocks in a CFG.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"

using namespace clang;

void PostOrderCFGView::anchor() {}

PostOrderCFGView::PostOrderCFGView(const CFG *cfg) {
  Blocks.reserve(cfg->getNumBlockIDs());

  // The CFG orders the blocks of loop bodies before those of loop successors
  // (both numerically, and in the successor order of the loop condition
  // block). So, RPO necessarily reverses that order, placing the loop successor
  // *before* the loop body. For many analyses, particularly those that converge
  // to a fixpoint, this results in potentially significant extra work because
  // loop successors will necessarily need to be reconsidered once the algorithm
  // has reached a fixpoint on the loop body.
  //
  // This definition of CFG graph traits reverses the order of children, so that
  // loop bodies will come first in an RPO.
  struct CFGLoopBodyFirstTraits {
    using NodeRef = const ::clang::CFGBlock *;
    using ChildIteratorType = ::clang::CFGBlock::const_succ_reverse_iterator;

    static ChildIteratorType child_begin(NodeRef N) { return N->succ_rbegin(); }
    static ChildIteratorType child_end(NodeRef N) { return N->succ_rend(); }
  };

  struct POTraversal
      : llvm::PostOrderTraversalBase<POTraversal, CFGLoopBodyFirstTraits> {
    CFGBlockSet BSet;

    POTraversal(const CFG *cfg) : BSet(cfg) { this->init(&cfg->getEntry()); }
    bool insertEdge(std::optional<const CFGBlock *>, const CFGBlock *To) {
      if (!To)
        return false;
      return BSet.insert(To).second;
    }
  };

  for (const CFGBlock *Block : POTraversal(cfg)) {
    BlockOrder[Block] = Blocks.size() + 1;
    Blocks.push_back(Block);
  }
}

std::unique_ptr<PostOrderCFGView>
PostOrderCFGView::create(AnalysisDeclContext &ctx) {
  const CFG *cfg = ctx.getCFG();
  if (!cfg)
    return nullptr;
  return std::make_unique<PostOrderCFGView>(cfg);
}

const void *PostOrderCFGView::getTag() { static int x; return &x; }

bool PostOrderCFGView::BlockOrderCompare::operator()(const CFGBlock *b1,
                                                     const CFGBlock *b2) const {
  PostOrderCFGView::BlockOrderTy::const_iterator b1It = POV.BlockOrder.find(b1);
  PostOrderCFGView::BlockOrderTy::const_iterator b2It = POV.BlockOrder.find(b2);

  unsigned b1V = (b1It == POV.BlockOrder.end()) ? 0 : b1It->second;
  unsigned b2V = (b2It == POV.BlockOrder.end()) ? 0 : b2It->second;
  return b1V > b2V;
}

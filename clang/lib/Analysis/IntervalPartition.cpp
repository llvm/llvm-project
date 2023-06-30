//===- IntervalPartition.cpp - CFG Partitioning into Intervals --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines functionality for partitioning a CFG into intervals.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/IntervalPartition.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/BitVector.h"
#include <queue>
#include <set>
#include <vector>

namespace clang {

static CFGInterval buildInterval(llvm::BitVector &Partitioned,
                                 const CFGBlock &Header) {
  CFGInterval Interval(&Header);
  Partitioned.set(Header.getBlockID());

  // Elements must not be null. Duplicates are prevented using `Workset`, below.
  std::queue<const CFGBlock *> Worklist;
  llvm::BitVector Workset(Header.getParent()->getNumBlockIDs(), false);
  for (const CFGBlock *S : Header.succs())
    if (S != nullptr)
      if (auto SID = S->getBlockID(); !Partitioned.test(SID)) {
        // Successors are unique, so we don't test against `Workset` before
        // adding to `Worklist`.
        Worklist.push(S);
        Workset.set(SID);
      }

  // Contains successors of blocks in the interval that couldn't be added to the
  // interval on their first encounter. This occurs when they have a predecessor
  // that is either definitively outside the interval or hasn't been considered
  // yet. In the latter case, we'll revisit the block through some other path
  // from the interval. At the end of processing the worklist, we filter out any
  // that ended up in the interval to produce the output set of interval
  // successors. It may contain duplicates -- ultimately, all relevant elements
  // are added to `Interval.Successors`, which is a set.
  std::vector<const CFGBlock *> MaybeSuccessors;

  while (!Worklist.empty()) {
    const auto *B = Worklist.front();
    auto ID = B->getBlockID();
    Worklist.pop();
    Workset.reset(ID);

    // Check whether all predecessors are in the interval, in which case `B`
    // is included as well.
    bool AllInInterval = true;
    for (const CFGBlock *P : B->preds())
      if (Interval.Blocks.find(P) == Interval.Blocks.end()) {
        MaybeSuccessors.push_back(B);
        AllInInterval = false;
        break;
      }
    if (AllInInterval) {
      Interval.Blocks.insert(B);
      Partitioned.set(ID);
      for (const CFGBlock *S : B->succs())
        if (S != nullptr)
          if (auto SID = S->getBlockID();
              !Partitioned.test(SID) && !Workset.test(SID)) {
            Worklist.push(S);
            Workset.set(SID);
          }
    }
  }

  // Any block successors not in the current interval are interval successors.
  for (const CFGBlock *B : MaybeSuccessors)
    if (Interval.Blocks.find(B) == Interval.Blocks.end())
      Interval.Successors.insert(B);

  return Interval;
}

CFGInterval buildInterval(const CFG &Cfg, const CFGBlock &Header) {
  llvm::BitVector Partitioned(Cfg.getNumBlockIDs(), false);
  return buildInterval(Partitioned, Header);
}

std::vector<CFGInterval> partitionIntoIntervals(const CFG &Cfg) {
  std::vector<CFGInterval> Intervals;
  llvm::BitVector Partitioned(Cfg.getNumBlockIDs(), false);
  auto &EntryBlock = Cfg.getEntry();
  Intervals.push_back(buildInterval(Partitioned, EntryBlock));

  std::queue<const CFGBlock *> Successors;
  for (const auto *S : Intervals[0].Successors)
    Successors.push(S);

  while (!Successors.empty()) {
    const auto *B = Successors.front();
    Successors.pop();
    if (Partitioned.test(B->getBlockID()))
      continue;

    // B has not been partitioned, but it has a predecessor that has.
    CFGInterval I = buildInterval(Partitioned, *B);
    for (const auto *S : I.Successors)
      Successors.push(S);
    Intervals.push_back(std::move(I));
  }

  return Intervals;
}

} // namespace clang

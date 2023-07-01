//===- IntervalPartition.h - CFG Partitioning into Intervals -----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines functionality for partitioning a CFG into intervals. The
//  concepts and implementations are based on the presentation in "Compilers" by
//  Aho, Sethi and Ullman (the "dragon book"), pages 664-666.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_INTERVALPARTITION_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_INTERVALPARTITION_H

#include "clang/Analysis/CFG.h"
#include "llvm/ADT/DenseSet.h"
#include <vector>

namespace clang {

// An interval is a strongly-connected component of the CFG along with a
// trailing acyclic structure. The _header_ of the interval is either the CFG
// entry block or has at least one predecessor outside of the interval. All
// other blocks in the interval have only predecessors also in the interval.
struct CFGInterval {
  CFGInterval(const CFGBlock *Header) : Header(Header), Blocks({Header}) {}

  // The block from which the interval was constructed. Is either the CFG entry
  // block or has at least one predecessor outside the interval.
  const CFGBlock *Header;

  llvm::SmallDenseSet<const CFGBlock *> Blocks;

  // Successor blocks of the *interval*: blocks outside the interval for
  // reachable (in one edge) from within the interval.
  llvm::SmallDenseSet<const CFGBlock *> Successors;
};

CFGInterval buildInterval(const CFG &Cfg, const CFGBlock &Header);

// Partitions `Cfg` into intervals and constructs a graph of the intervals,
// based on the edges between nodes in these intervals.
std::vector<CFGInterval> partitionIntoIntervals(const CFG &Cfg);

} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_INTERVALPARTITION_H

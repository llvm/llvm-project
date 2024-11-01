//===- bolt/Passes/MCF.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions for solving minimum-cost flow problem.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/MCF.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Passes/DataflowInfoManager.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
#include <vector>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "mcf"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

extern cl::opt<bool> TimeOpts;

static cl::opt<bool> IterativeGuess(
    "iterative-guess",
    cl::desc("in non-LBR mode, guess edge counts using iterative technique"),
    cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool> UseRArcs(
    "mcf-use-rarcs",
    cl::desc("in MCF, consider the possibility of cancelling flow to balance "
             "edges"),
    cl::Hidden, cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {

namespace {

// Edge Weight Inference Heuristic
//
// We start by maintaining the invariant used in LBR mode where the sum of
// pred edges count is equal to the block execution count. This loop will set
// pred edges count by balancing its own execution count in different pred
// edges. The weight of each edge is guessed by looking at how hot each pred
// block is (in terms of samples).
// There are two caveats in this approach. One is for critical edges and the
// other is for self-referencing blocks (loops of 1 BB). For critical edges,
// we can't infer the hotness of them based solely on pred BBs execution
// count. For each critical edge we look at the pred BB, then look at its
// succs to adjust its weight.
//
//    [ 60  ]       [ 25 ]
//       |      \     |
//    [ 10  ]       [ 75 ]
//
// The illustration above shows a critical edge \. We wish to adjust bb count
// 60 to 50 to properly determine the weight of the critical edge to be
// 50 / 75.
// For self-referencing edges, we attribute its weight by subtracting the
// current BB execution count by the sum of predecessors count if this result
// is non-negative.
using EdgeWeightMap =
    DenseMap<std::pair<const BinaryBasicBlock *, const BinaryBasicBlock *>,
             double>;

template <class NodeT>
void updateEdgeWeight(EdgeWeightMap &EdgeWeights, const BinaryBasicBlock *A,
                      const BinaryBasicBlock *B, double Weight);

template <>
void updateEdgeWeight<BinaryBasicBlock *>(EdgeWeightMap &EdgeWeights,
                                          const BinaryBasicBlock *A,
                                          const BinaryBasicBlock *B,
                                          double Weight) {
  EdgeWeights[std::make_pair(A, B)] = Weight;
}

template <>
void updateEdgeWeight<Inverse<BinaryBasicBlock *>>(EdgeWeightMap &EdgeWeights,
                                                   const BinaryBasicBlock *A,
                                                   const BinaryBasicBlock *B,
                                                   double Weight) {
  EdgeWeights[std::make_pair(B, A)] = Weight;
}

template <class NodeT>
void computeEdgeWeights(BinaryBasicBlock *BB, EdgeWeightMap &EdgeWeights) {
  typedef GraphTraits<NodeT> GraphT;
  typedef GraphTraits<Inverse<NodeT>> InvTraits;

  double TotalChildrenCount = 0.0;
  SmallVector<double, 4> ChildrenExecCount;
  // First pass computes total children execution count that directly
  // contribute to this BB.
  for (typename GraphT::ChildIteratorType CI = GraphT::child_begin(BB),
                                          E = GraphT::child_end(BB);
       CI != E; ++CI) {
    typename GraphT::NodeRef Child = *CI;
    double ChildExecCount = Child->getExecutionCount();
    // Is self-reference?
    if (Child == BB) {
      ChildExecCount = 0.0; // will fill this in second pass
    } else if (GraphT::child_end(BB) - GraphT::child_begin(BB) > 1 &&
               InvTraits::child_end(Child) - InvTraits::child_begin(Child) >
                   1) {
      // Handle critical edges. This will cause a skew towards crit edges, but
      // it is a quick solution.
      double CritWeight = 0.0;
      uint64_t Denominator = 0;
      for (typename InvTraits::ChildIteratorType
               II = InvTraits::child_begin(Child),
               IE = InvTraits::child_end(Child);
           II != IE; ++II) {
        typename GraphT::NodeRef N = *II;
        Denominator += N->getExecutionCount();
        if (N != BB)
          continue;
        CritWeight = N->getExecutionCount();
      }
      if (Denominator)
        CritWeight /= static_cast<double>(Denominator);
      ChildExecCount *= CritWeight;
    }
    ChildrenExecCount.push_back(ChildExecCount);
    TotalChildrenCount += ChildExecCount;
  }
  // Second pass fixes the weight of a possible self-reference edge
  uint32_t ChildIndex = 0;
  for (typename GraphT::ChildIteratorType CI = GraphT::child_begin(BB),
                                          E = GraphT::child_end(BB);
       CI != E; ++CI) {
    typename GraphT::NodeRef Child = *CI;
    if (Child != BB) {
      ++ChildIndex;
      continue;
    }
    if (static_cast<double>(BB->getExecutionCount()) > TotalChildrenCount) {
      ChildrenExecCount[ChildIndex] =
          BB->getExecutionCount() - TotalChildrenCount;
      TotalChildrenCount += ChildrenExecCount[ChildIndex];
    }
    break;
  }
  // Third pass finally assigns weights to edges
  ChildIndex = 0;
  for (typename GraphT::ChildIteratorType CI = GraphT::child_begin(BB),
                                          E = GraphT::child_end(BB);
       CI != E; ++CI) {
    typename GraphT::NodeRef Child = *CI;
    double Weight = 1 / (GraphT::child_end(BB) - GraphT::child_begin(BB));
    if (TotalChildrenCount != 0.0)
      Weight = ChildrenExecCount[ChildIndex] / TotalChildrenCount;
    updateEdgeWeight<NodeT>(EdgeWeights, BB, Child, Weight);
    ++ChildIndex;
  }
}

template <class NodeT>
void computeEdgeWeights(BinaryFunction &BF, EdgeWeightMap &EdgeWeights) {
  for (BinaryBasicBlock &BB : BF)
    computeEdgeWeights<NodeT>(&BB, EdgeWeights);
}

/// Make BB count match the sum of all incoming edges. If AllEdges is true,
/// make it match max(SumPredEdges, SumSuccEdges).
void recalculateBBCounts(BinaryFunction &BF, bool AllEdges) {
  for (BinaryBasicBlock &BB : BF) {
    uint64_t TotalPredsEWeight = 0;
    for (BinaryBasicBlock *Pred : BB.predecessors())
      TotalPredsEWeight += Pred->getBranchInfo(BB).Count;

    if (TotalPredsEWeight > BB.getExecutionCount())
      BB.setExecutionCount(TotalPredsEWeight);

    if (!AllEdges)
      continue;

    uint64_t TotalSuccsEWeight = 0;
    for (BinaryBasicBlock::BinaryBranchInfo &BI : BB.branch_info())
      TotalSuccsEWeight += BI.Count;

    if (TotalSuccsEWeight > BB.getExecutionCount())
      BB.setExecutionCount(TotalSuccsEWeight);
  }
}

// This is our main edge count guessing heuristic. Look at predecessors and
// assign a proportionally higher count to pred edges coming from blocks with
// a higher execution count in comparison with the other predecessor blocks,
// making SumPredEdges match the current BB count.
// If "UseSucc" is true, apply the same logic to successor edges as well. Since
// some successor edges may already have assigned a count, only update it if the
// new count is higher.
void guessEdgeByRelHotness(BinaryFunction &BF, bool UseSucc,
                           EdgeWeightMap &PredEdgeWeights,
                           EdgeWeightMap &SuccEdgeWeights) {
  for (BinaryBasicBlock &BB : BF) {
    for (BinaryBasicBlock *Pred : BB.predecessors()) {
      double RelativeExec = PredEdgeWeights[std::make_pair(Pred, &BB)];
      RelativeExec *= BB.getExecutionCount();
      BinaryBasicBlock::BinaryBranchInfo &BI = Pred->getBranchInfo(BB);
      if (static_cast<uint64_t>(RelativeExec) > BI.Count)
        BI.Count = static_cast<uint64_t>(RelativeExec);
    }

    if (!UseSucc)
      continue;

    auto BI = BB.branch_info_begin();
    for (BinaryBasicBlock *Succ : BB.successors()) {
      double RelativeExec = SuccEdgeWeights[std::make_pair(&BB, Succ)];
      RelativeExec *= BB.getExecutionCount();
      if (static_cast<uint64_t>(RelativeExec) > BI->Count)
        BI->Count = static_cast<uint64_t>(RelativeExec);
      ++BI;
    }
  }
}

using ArcSet =
    DenseSet<std::pair<const BinaryBasicBlock *, const BinaryBasicBlock *>>;

/// Predecessor edges version of guessEdgeByIterativeApproach. GuessedArcs has
/// all edges we already established their count. Try to guess the count of
/// the remaining edge, if there is only one to guess, and return true if we
/// were able to guess.
bool guessPredEdgeCounts(BinaryBasicBlock *BB, ArcSet &GuessedArcs) {
  if (BB->pred_size() == 0)
    return false;

  uint64_t TotalPredCount = 0;
  unsigned NumGuessedEdges = 0;
  for (BinaryBasicBlock *Pred : BB->predecessors()) {
    if (GuessedArcs.count(std::make_pair(Pred, BB)))
      ++NumGuessedEdges;
    TotalPredCount += Pred->getBranchInfo(*BB).Count;
  }

  if (NumGuessedEdges != BB->pred_size() - 1)
    return false;

  int64_t Guessed =
      static_cast<int64_t>(BB->getExecutionCount()) - TotalPredCount;
  if (Guessed < 0)
    Guessed = 0;

  for (BinaryBasicBlock *Pred : BB->predecessors()) {
    if (GuessedArcs.count(std::make_pair(Pred, BB)))
      continue;

    Pred->getBranchInfo(*BB).Count = Guessed;
    return true;
  }
  llvm_unreachable("Expected unguessed arc");
}

/// Successor edges version of guessEdgeByIterativeApproach. GuessedArcs has
/// all edges we already established their count. Try to guess the count of
/// the remaining edge, if there is only one to guess, and return true if we
/// were able to guess.
bool guessSuccEdgeCounts(BinaryBasicBlock *BB, ArcSet &GuessedArcs) {
  if (BB->succ_size() == 0)
    return false;

  uint64_t TotalSuccCount = 0;
  unsigned NumGuessedEdges = 0;
  auto BI = BB->branch_info_begin();
  for (BinaryBasicBlock *Succ : BB->successors()) {
    if (GuessedArcs.count(std::make_pair(BB, Succ)))
      ++NumGuessedEdges;
    TotalSuccCount += BI->Count;
    ++BI;
  }

  if (NumGuessedEdges != BB->succ_size() - 1)
    return false;

  int64_t Guessed =
      static_cast<int64_t>(BB->getExecutionCount()) - TotalSuccCount;
  if (Guessed < 0)
    Guessed = 0;

  BI = BB->branch_info_begin();
  for (BinaryBasicBlock *Succ : BB->successors()) {
    if (GuessedArcs.count(std::make_pair(BB, Succ))) {
      ++BI;
      continue;
    }

    BI->Count = Guessed;
    GuessedArcs.insert(std::make_pair(BB, Succ));
    return true;
  }
  llvm_unreachable("Expected unguessed arc");
}

/// Guess edge count whenever we have only one edge (pred or succ) left
/// to guess. Then make its count equal to BB count minus all other edge
/// counts we already know their count. Repeat this until there is no
/// change.
void guessEdgeByIterativeApproach(BinaryFunction &BF) {
  ArcSet KnownArcs;
  bool Changed = false;

  do {
    Changed = false;
    for (BinaryBasicBlock &BB : BF) {
      if (guessPredEdgeCounts(&BB, KnownArcs))
        Changed = true;
      if (guessSuccEdgeCounts(&BB, KnownArcs))
        Changed = true;
    }
  } while (Changed);

  // Guess count for non-inferred edges
  for (BinaryBasicBlock &BB : BF) {
    for (BinaryBasicBlock *Pred : BB.predecessors()) {
      if (KnownArcs.count(std::make_pair(Pred, &BB)))
        continue;
      BinaryBasicBlock::BinaryBranchInfo &BI = Pred->getBranchInfo(BB);
      BI.Count =
          std::min(Pred->getExecutionCount(), BB.getExecutionCount()) / 2;
      KnownArcs.insert(std::make_pair(Pred, &BB));
    }
    auto BI = BB.branch_info_begin();
    for (BinaryBasicBlock *Succ : BB.successors()) {
      if (KnownArcs.count(std::make_pair(&BB, Succ))) {
        ++BI;
        continue;
      }
      BI->Count =
          std::min(BB.getExecutionCount(), Succ->getExecutionCount()) / 2;
      KnownArcs.insert(std::make_pair(&BB, Succ));
      break;
    }
  }
}

/// Associate each basic block with the BinaryLoop object corresponding to the
/// innermost loop containing this block.
DenseMap<const BinaryBasicBlock *, const BinaryLoop *>
createLoopNestLevelMap(BinaryFunction &BF) {
  DenseMap<const BinaryBasicBlock *, const BinaryLoop *> LoopNestLevel;
  const BinaryLoopInfo &BLI = BF.getLoopInfo();

  for (BinaryBasicBlock &BB : BF)
    LoopNestLevel[&BB] = BLI[&BB];

  return LoopNestLevel;
}

} // end anonymous namespace

void equalizeBBCounts(DataflowInfoManager &Info, BinaryFunction &BF) {
  if (BF.begin() == BF.end())
    return;

  DominatorAnalysis<false> &DA = Info.getDominatorAnalysis();
  DominatorAnalysis<true> &PDA = Info.getPostDominatorAnalysis();
  auto &InsnToBB = Info.getInsnToBBMap();
  // These analyses work at the instruction granularity, but we really only need
  // basic block granularity here. So we'll use a set of visited edges to avoid
  // revisiting the same BBs again and again.
  DenseMap<const BinaryBasicBlock *, std::set<const BinaryBasicBlock *>>
      Visited;
  // Equivalence classes mapping. Each equivalence class is defined by the set
  // of BBs that obeys the aforementioned properties.
  DenseMap<const BinaryBasicBlock *, signed> BBsToEC;
  std::vector<std::vector<BinaryBasicBlock *>> Classes;

  BF.calculateLoopInfo();
  DenseMap<const BinaryBasicBlock *, const BinaryLoop *> LoopNestLevel =
      createLoopNestLevelMap(BF);

  for (BinaryBasicBlock &BB : BF)
    BBsToEC[&BB] = -1;

  for (BinaryBasicBlock &BB : BF) {
    auto I = BB.begin();
    if (I == BB.end())
      continue;

    DA.doForAllDominators(*I, [&](const MCInst &DomInst) {
      BinaryBasicBlock *DomBB = InsnToBB[&DomInst];
      if (Visited[DomBB].count(&BB))
        return;
      Visited[DomBB].insert(&BB);
      if (!PDA.doesADominateB(*I, DomInst))
        return;
      if (LoopNestLevel[&BB] != LoopNestLevel[DomBB])
        return;
      if (BBsToEC[DomBB] == -1 && BBsToEC[&BB] == -1) {
        BBsToEC[DomBB] = Classes.size();
        BBsToEC[&BB] = Classes.size();
        Classes.emplace_back();
        Classes.back().push_back(DomBB);
        Classes.back().push_back(&BB);
        return;
      }
      if (BBsToEC[DomBB] == -1) {
        BBsToEC[DomBB] = BBsToEC[&BB];
        Classes[BBsToEC[&BB]].push_back(DomBB);
        return;
      }
      if (BBsToEC[&BB] == -1) {
        BBsToEC[&BB] = BBsToEC[DomBB];
        Classes[BBsToEC[DomBB]].push_back(&BB);
        return;
      }
      signed BBECNum = BBsToEC[&BB];
      std::vector<BinaryBasicBlock *> DomEC = Classes[BBsToEC[DomBB]];
      std::vector<BinaryBasicBlock *> BBEC = Classes[BBECNum];
      for (BinaryBasicBlock *Block : DomEC) {
        BBsToEC[Block] = BBECNum;
        BBEC.push_back(Block);
      }
      DomEC.clear();
    });
  }

  for (std::vector<BinaryBasicBlock *> &Class : Classes) {
    uint64_t Max = 0ULL;
    for (BinaryBasicBlock *BB : Class)
      Max = std::max(Max, BB->getExecutionCount());
    for (BinaryBasicBlock *BB : Class)
      BB->setExecutionCount(Max);
  }
}

void estimateEdgeCounts(BinaryFunction &BF) {
  EdgeWeightMap PredEdgeWeights;
  EdgeWeightMap SuccEdgeWeights;
  if (!opts::IterativeGuess) {
    computeEdgeWeights<Inverse<BinaryBasicBlock *>>(BF, PredEdgeWeights);
    computeEdgeWeights<BinaryBasicBlock *>(BF, SuccEdgeWeights);
  }
  if (opts::EqualizeBBCounts) {
    LLVM_DEBUG(BF.print(dbgs(), "before equalize BB counts"));
    auto Info = DataflowInfoManager(BF, nullptr, nullptr);
    equalizeBBCounts(Info, BF);
    LLVM_DEBUG(BF.print(dbgs(), "after equalize BB counts"));
  }
  if (opts::IterativeGuess)
    guessEdgeByIterativeApproach(BF);
  else
    guessEdgeByRelHotness(BF, /*UseSuccs=*/false, PredEdgeWeights,
                          SuccEdgeWeights);
  recalculateBBCounts(BF, /*AllEdges=*/false);
}

void solveMCF(BinaryFunction &BF, MCFCostFunction CostFunction) {
  llvm_unreachable("not implemented");
}

} // namespace bolt
} // namespace llvm

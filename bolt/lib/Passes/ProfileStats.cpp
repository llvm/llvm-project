//===- bolt/Passes/ProfileStats.cpp - profile quality metrics ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions to print profile stats to quantify profile quality.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/ProfileStats.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>

#define DEBUG_TYPE "bolt-opts"

using namespace llvm;
using namespace bolt;

namespace opts {
extern cl::opt<unsigned> Verbosity;
cl::opt<bool> PrintBucketedMetrics(
    "print-bucketed-profile-stats",
    cl::desc("print profile quality stats for buckets of functions created "
             "based on their execution counts."),
    cl::Hidden, cl::cat(BoltCategory));
cl::opt<unsigned>
    NumFunctionsPerBucket("num-functions-per-bucket",
                          cl::desc("Maximum number of functions per bucket."),
                          cl::init(500), cl::ZeroOrMore, cl::Hidden,
                          cl::cat(BoltOptCategory));
cl::opt<unsigned> NumTopFunctions(
    "num-top-functions",
    cl::desc(
        "Number of hottest functions to print aggregated profile stats of."),
    cl::init(1000), cl::ZeroOrMore, cl::Hidden, cl::cat(BoltOptCategory));
cl::opt<unsigned>
    BBECThreshold("bbec-threshold",
                  cl::desc("Minimum execution count of a basic block for it to "
                           "be considered for profile stats computation."),
                  cl::init(100), cl::ZeroOrMore, cl::Hidden,
                  cl::cat(BoltOptCategory));
} // namespace opts

namespace {
void printProfileBiasScore(raw_ostream &OS, BinaryContext &BC) {
  double FlowImbalanceMean = 0.0;
  size_t NumBlocksConsidered = 0;
  double WorstBias = 0.0;
  const BinaryFunction *WorstBiasFunc = nullptr;

  // For each function CFG, we fill an IncomingMap with the sum of the frequency
  // of incoming edges for each BB. Likewise for each OutgoingMap and the sum
  // of the frequency of outgoing edges.
  using FlowMapTy = std::unordered_map<const BinaryBasicBlock *, uint64_t>;
  std::unordered_map<const BinaryFunction *, FlowMapTy> TotalIncomingMaps;
  std::unordered_map<const BinaryFunction *, FlowMapTy> TotalOutgoingMaps;

  // Compute mean
  for (const auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction &Function = BFI.second;
    if (Function.empty() || !Function.isSimple())
      continue;
    FlowMapTy &IncomingMap = TotalIncomingMaps[&Function];
    FlowMapTy &OutgoingMap = TotalOutgoingMaps[&Function];
    for (const BinaryBasicBlock &BB : Function) {
      uint64_t TotalOutgoing = 0ULL;
      auto SuccBIIter = BB.branch_info_begin();
      for (BinaryBasicBlock *Succ : BB.successors()) {
        uint64_t Count = SuccBIIter->Count;
        if (Count == BinaryBasicBlock::COUNT_NO_PROFILE || Count == 0) {
          ++SuccBIIter;
          continue;
        }
        TotalOutgoing += Count;
        IncomingMap[Succ] += Count;
        ++SuccBIIter;
      }
      OutgoingMap[&BB] = TotalOutgoing;
    }

    size_t NumBlocks = 0;
    double Mean = 0.0;
    for (const BinaryBasicBlock &BB : Function) {
      // Do not compute score for low frequency blocks, entry or exit blocks
      if (IncomingMap[&BB] < 100 || OutgoingMap[&BB] == 0 || BB.isEntryPoint())
        continue;
      ++NumBlocks;
      const double Difference = (double)OutgoingMap[&BB] - IncomingMap[&BB];
      Mean += fabs(Difference / IncomingMap[&BB]);
    }

    FlowImbalanceMean += Mean;
    NumBlocksConsidered += NumBlocks;
    if (!NumBlocks)
      continue;
    double FuncMean = Mean / NumBlocks;
    if (FuncMean > WorstBias) {
      WorstBias = FuncMean;
      WorstBiasFunc = &Function;
    }
  }
  if (NumBlocksConsidered > 0)
    FlowImbalanceMean /= NumBlocksConsidered;

  // Compute standard deviation
  NumBlocksConsidered = 0;
  double FlowImbalanceVar = 0.0;
  for (const auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction &Function = BFI.second;
    if (Function.empty() || !Function.isSimple())
      continue;
    FlowMapTy &IncomingMap = TotalIncomingMaps[&Function];
    FlowMapTy &OutgoingMap = TotalOutgoingMaps[&Function];
    for (const BinaryBasicBlock &BB : Function) {
      if (IncomingMap[&BB] < 100 || OutgoingMap[&BB] == 0)
        continue;
      ++NumBlocksConsidered;
      const double Difference = (double)OutgoingMap[&BB] - IncomingMap[&BB];
      FlowImbalanceVar +=
          pow(fabs(Difference / IncomingMap[&BB]) - FlowImbalanceMean, 2);
    }
  }
  if (NumBlocksConsidered) {
    FlowImbalanceVar /= NumBlocksConsidered;
    FlowImbalanceVar = sqrt(FlowImbalanceVar);
  }

  // Report to user
  OS << format("BOLT-INFO: Profile bias score: %.4lf%% StDev: %.4lf%%\n",
               (100.0 * FlowImbalanceMean), (100.0 * FlowImbalanceVar));
  if (WorstBiasFunc && opts::Verbosity >= 1) {
    OS << "Worst average bias observed in " << WorstBiasFunc->getPrintName()
       << "\n";
    LLVM_DEBUG(WorstBiasFunc->dump());
  }
}

using FunctionListType = std::vector<const BinaryFunction *>;
using function_iterator = FunctionListType::iterator;
using FlowMapTy = std::unordered_map<const BinaryBasicBlock *, uint64_t>;
using TotalFlowMapTy = std::unordered_map<const BinaryFunction *, FlowMapTy>;
using FunctionFlowMapTy = std::unordered_map<const BinaryFunction *, uint64_t>;

struct FlowInfo {
  TotalFlowMapTy TotalIncomingMaps;
  TotalFlowMapTy TotalOutgoingMaps;
  TotalFlowMapTy TotalIECMaps;
  TotalFlowMapTy TotalMaxCallMaps;
  FunctionFlowMapTy CallGraphIncomingMap;
};

template <typename T>
void printDistribution(raw_ostream &OS, std::vector<T> &values,
                       bool Fraction = false) {
  if (values.empty())
    return;
  // Sort values from largest to smallest and print the MAX, TOP 1%, 5%, 10%,
  // 20%, 50%, 80%, MIN. If Fraction is true, then values are printed as
  // fractions instead of integers.
  std::sort(values.begin(), values.end());

  auto printLine = [&](std::string Text, double Percent) {
    int Rank = int(values.size() * (1.0 - Percent / 100));
    if (Percent == 0)
      Rank = values.size() - 1;
    if (Fraction)
      OS << "  " << Text << std::string(9 - Text.length(), ' ') << ": "
         << format("%.2lf%%", values[Rank] * 100) << "\n";
    else
      OS << "  " << Text << std::string(9 - Text.length(), ' ') << ": "
         << values[Rank] << "\n";
  };

  printLine("MAX", 0);
  int percentages[] = {1, 5, 10, 20, 50, 80};
  for (size_t i = 0; i < sizeof(percentages) / sizeof(percentages[0]); ++i) {
    printLine("TOP " + std::to_string(percentages[i]) + "%", percentages[i]);
  }
  printLine("MIN", 100);
}

void printKECMetrics(raw_ostream &OS,
                     iterator_range<function_iterator> &Functions,
                     size_t BBECThreshold, FlowInfo &TotalFlowMap) {
  // Each BB's Inferred Execution Count (IEC) equals to the max of its
  // inflow, outflow, and individual call counts (if any).
  // For each BB, its KEC = BinaryBasicBlock::getKnownExecutionCount() should be
  // equal to its IEC in a perfect profile.
  TotalFlowMapTy &TotalIECMaps = TotalFlowMap.TotalIECMaps;
  size_t NumConsideredBBs = 0;
  std::vector<size_t> PosDiffs;
  for (auto it = Functions.begin(); it != Functions.end(); ++it) {
    const BinaryFunction *Function = *it;
    if (Function->size() <= 1)
      continue;
    FlowMapTy &IECMap = TotalIECMaps[Function];
    for (const BinaryBasicBlock &BB : *Function) {
      size_t IEC = IECMap[&BB];
      size_t KEC = BB.getKnownExecutionCount();
      size_t BBEC = std::max(IEC, KEC);
      // Do not consider low frequency blocks.
      // A low frequency block is a block whose max(IEC, KEC) is below a given
      // threshold.
      if (BBEC < BBECThreshold)
        continue;
      NumConsideredBBs++;
      if (IEC <= KEC)
        continue;
      PosDiffs.push_back(IEC - KEC);
    }
  }
  OS << "-------------------------------------------------------\n"
     << "Metric 1: KEC <> IEC\n"
     << "-------------------------------------------------------\n";
  if (NumConsideredBBs == 0) {
    OS << "  No BBs considered for this metric.\n";
  } else {
    OS << format("- %zu (%.2lf%%) of considered BBs have IEC > KEC\n",
                 PosDiffs.size(),
                 100.0 * (double)PosDiffs.size() / NumConsideredBBs);

    if (!PosDiffs.empty()) {
      OS << "- Distribution of (IEC - KEC) among considered BBs with IEC > "
            "KEC\n";
      printDistribution(OS, PosDiffs);
    }
  }
}

void printFlowConservationMetrics(raw_ostream &OS,
                                  iterator_range<function_iterator> &Functions,
                                  size_t BBECThreshold,
                                  FlowInfo &TotalFlowMap) {
  // Each non-entry non-exit BB should have its inflow equal to its outflow in a
  // perfect profile.
  TotalFlowMapTy &TotalIncomingMaps = TotalFlowMap.TotalIncomingMaps;
  TotalFlowMapTy &TotalOutgoingMaps = TotalFlowMap.TotalOutgoingMaps;
  TotalFlowMapTy &TotalIECMaps = TotalFlowMap.TotalIECMaps;

  size_t NumConsideredBBs = 0;
  size_t NumFocalBBs = 0;
  std::vector<double> MaxDiffFracs;
  std::vector<size_t> MaxDiffs;
  for (auto it = Functions.begin(); it != Functions.end(); ++it) {
    const BinaryFunction *Function = *it;
    if (Function->size() <= 1)
      continue;
    FlowMapTy &IncomingMap = TotalIncomingMaps[Function];
    FlowMapTy &OutgoingMap = TotalOutgoingMaps[Function];
    FlowMapTy &IECMap = TotalIECMaps[Function];

    size_t MaxDifference = 0.0;
    double MaxDiffFrac = 0.0;
    for (const BinaryBasicBlock &BB : *Function) {
      size_t BBEC = std::max(IECMap[&BB], BB.getKnownExecutionCount());
      // Do not consider low frequency blocks.
      if (BBEC < BBECThreshold)
        continue;
      NumConsideredBBs++;
      // Do not consider entry or exit blocks.
      if (BB.succ_size() == 0 || BB.isEntryPoint())
        continue;
      NumFocalBBs++;

      size_t LHS = IncomingMap[&BB];
      size_t RHS = OutgoingMap[&BB];

      size_t MIN = std::min(LHS, RHS);
      size_t MAX = std::max(LHS, RHS);
      const size_t Difference = MAX - MIN;
      double DiffFrac = 0.0;
      if (MAX > 0)
        DiffFrac = (double)Difference / MAX;
      if (DiffFrac > MaxDiffFrac)
        MaxDiffFrac = DiffFrac;
      if (Difference > MaxDifference)
        MaxDifference = Difference;
      if (opts::Verbosity >= 2 && DiffFrac > 0.5 && Difference > 500) {
        OS << "Big flow conservation violation observed in " << BB.getName()
           << " in function " << Function->getPrintName() << "\n";
        LLVM_DEBUG(Function->dump());
      }
    }
    if (NumFocalBBs > 0) {
      MaxDiffFracs.push_back(MaxDiffFrac);
      MaxDiffs.push_back(MaxDifference);
    }
  }

  OS << "-------------------------------------------------------\n"
     << "Metric 2: BB inflow <> BB outflow\n"
     << "-------------------------------------------------------\n";
  if (NumFocalBBs == 0) {
    OS << "  No BBs considered for this metric.\n";
  } else {
    OS << format("Focus on %zu (%.2lf%%) of considered BBs that are neither "
                 "function entries nor exits\n",
                 NumFocalBBs, 100.0 * (double)NumFocalBBs / NumConsideredBBs)
       << format("Focus on %zu (%.2lf%%) of considered functions that has at "
                 "least 1 focal BBs\n",
                 MaxDiffs.size(),
                 100.0 * (double)MaxDiffs.size() /
                     (std::distance(Functions.begin(), Functions.end())))
       << "LHS = BB SUM inflow; RHS = BB SUM outflow\n"
       << "MAX = MAX(LHS, RHS); MIN = MIN(LHS, RHS)\n";

    if (!MaxDiffFracs.empty()) {
      OS << "- Distribution of Worst[(MAX - MIN) / MAX] among all considered "
            "functions\n";
      printDistribution(OS, MaxDiffFracs, /*Fraction*/ true);
    }
    if (!MaxDiffs.empty()) {
      OS << "- Distribution of Worst[MAX - MIN] among all considered "
            "functions\n";
      printDistribution(OS, MaxDiffs);
    }
  }
}

void printFlowCallConservationMetrics(
    raw_ostream &OS, iterator_range<function_iterator> &Functions,
    size_t BBECThreshold, FlowInfo &TotalFlowMap) {
  // Each non-entry non-exit BB that makes function call(s) should have its
  // max(inflow, outflow) equal to its call count(s) in a perfect profile.
  TotalFlowMapTy &TotalIncomingMaps = TotalFlowMap.TotalIncomingMaps;
  TotalFlowMapTy &TotalOutgoingMaps = TotalFlowMap.TotalOutgoingMaps;
  TotalFlowMapTy &TotalIECMaps = TotalFlowMap.TotalIECMaps;
  TotalFlowMapTy &TotalMaxCallMaps = TotalFlowMap.TotalMaxCallMaps;

  size_t NumConsideredBBs = 0;
  size_t NumFocalBBs = 0;
  std::vector<double> MaxDiffFracs;
  std::vector<size_t> MaxDiffs;
  for (auto it = Functions.begin(); it != Functions.end(); ++it) {
    const BinaryFunction *Function = *it;
    if (Function->size() <= 1)
      continue;
    FlowMapTy &IncomingMap = TotalIncomingMaps[Function];
    FlowMapTy &OutgoingMap = TotalOutgoingMaps[Function];
    FlowMapTy &IECMap = TotalIECMaps[Function];
    FlowMapTy &MaxCallMap = TotalMaxCallMaps[Function];

    size_t MaxDifference = 0.0;
    double MaxDiffFrac = 0.0;
    bool FunctionConsidered = false;
    for (const BinaryBasicBlock &BB : *Function) {
      size_t BBEC = std::max(IECMap[&BB], BB.getKnownExecutionCount());
      // Do not consider low frequency blocks.
      if (BBEC < BBECThreshold)
        continue;
      NumConsideredBBs++;
      // Do not consider BBs that do not make function calls.
      if (MaxCallMap.find(&BB) == MaxCallMap.end())
        continue;
      NumFocalBBs++;
      FunctionConsidered = true;

      size_t LHS = std::max(IncomingMap[&BB], OutgoingMap[&BB]);
      size_t RHS = MaxCallMap[&BB];

      size_t MIN = std::min(LHS, RHS);
      size_t MAX = std::max(LHS, RHS);
      const size_t Difference = MAX - MIN;
      double DiffFrac = 0.0;
      if (MAX > 0)
        DiffFrac = (double)Difference / MAX;
      if (DiffFrac > MaxDiffFrac)
        MaxDiffFrac = DiffFrac;
      if (Difference > MaxDifference)
        MaxDifference = Difference;
      if (opts::Verbosity >= 2 && DiffFrac > 0.5 && Difference > 500) {
        OS << "Big flow call count conservation violation observed in "
           << BB.getName() << " in function " << Function->getPrintName()
           << "\n";
        LLVM_DEBUG(Function->dump());
      }
    }
    if (FunctionConsidered) {
      MaxDiffFracs.push_back(MaxDiffFrac);
      MaxDiffs.push_back(MaxDifference);
    }
  }
  OS << "-------------------------------------------------------\n"
     << "Metric 3: BB flows <> BB call ECs\n"
     << "-------------------------------------------------------\n";
  if (NumFocalBBs == 0) {
    OS << "  No BBs considered for this metric.\n";
  } else {
    OS << format("Focus on %zu (%.2lf%%) of considered BBs that make at least "
                 "one function call\n",
                 NumFocalBBs, 100.0 * (double)NumFocalBBs / NumConsideredBBs)
       << format("Focus on %zu (%.2lf%%) of considered functions that has at "
                 "least 1 focal BBs\n",
                 MaxDiffs.size(),
                 100.0 * (double)MaxDiffs.size() /
                     (std::distance(Functions.begin(), Functions.end())))
       << "LHS = MAX(BB SUM inflow, BB SUM outflow); RHS = MAX(BB call "
          "counts)\n"
       << "MAX = MAX(LHS, RHS); MIN = MIN(LHS, RHS)\n";

    if (!MaxDiffFracs.empty()) {
      OS << "- Distribution of Worst[(MAX - MIN) / MAX] among all considered "
            "functions\n";
      printDistribution(OS, MaxDiffFracs, /*Fraction*/ true);
    }
    if (!MaxDiffs.empty()) {
      OS << "- Distribution of Worst[MAX - MIN] among all considered "
            "functions\n";
      printDistribution(OS, MaxDiffs);
    }
  }
}

void printCallGraphConsistencyMetrics(
    raw_ostream &OS, iterator_range<function_iterator> &Functions,
    FlowInfo &TotalFlowMap) {
  // For every non-program-entry function, the total number of calls to this
  // function should be equal to its net entry outflow in a perfect profile.
  TotalFlowMapTy &TotalIncomingMaps = TotalFlowMap.TotalIncomingMaps;
  TotalFlowMapTy &TotalOutgoingMaps = TotalFlowMap.TotalOutgoingMaps;
  FunctionFlowMapTy &CallGraphIncomingMap = TotalFlowMap.CallGraphIncomingMap;

  std::vector<uint64_t> AbsDiffs;
  std::vector<double> FractionDiffs;

  for (auto it = Functions.begin(); it != Functions.end(); ++it) {
    const BinaryFunction *Function = *it;

    // If Function is not in CallGraphIncomingMap, then it is not called by any
    // other functions and likely a program entry.
    if (CallGraphIncomingMap.find(Function) == CallGraphIncomingMap.end())
      continue;

    FlowMapTy &IncomingMap = TotalIncomingMaps[Function];
    FlowMapTy &OutgoingMap = TotalOutgoingMaps[Function];

    uint64_t EntryIncoming = 0ULL;
    uint64_t EntryOutgoing = 0ULL;

    bool HasExitEntryBB = false;
    for (const BinaryBasicBlock &BB : *Function) {
      if (!BB.isEntryPoint())
        continue;
      if (BB.succ_size() == 0) {
        HasExitEntryBB = true;
        break;
      }
      EntryIncoming += IncomingMap[&BB];
      EntryOutgoing += OutgoingMap[&BB];
    }
    // If any entry BB of the current function is an exit as well, then we don't
    // consider it because the conservation doesn't need to hold.
    if (HasExitEntryBB)
      continue;

    assert(EntryIncoming <= EntryOutgoing);
    uint64_t RHS = EntryOutgoing - EntryIncoming;
    uint64_t LHS = CallGraphIncomingMap[Function];

    size_t MIN = std::min(LHS, RHS);
    size_t MAX = std::max(LHS, RHS);
    const size_t Difference = MAX - MIN;
    double DiffFrac = 0.0;
    if (MAX > 0)
      DiffFrac = (double)Difference / MAX;
    if (opts::Verbosity >= 2 && DiffFrac > 0.5 && Difference > 500) {
      OS << "Big function call inflow flow net entry outflow conservation "
            "violation observed in "
         << "function " << Function->getPrintName() << "\n";
      LLVM_DEBUG(Function->dump());
    }
    AbsDiffs.push_back(Difference);
    FractionDiffs.push_back(DiffFrac);
  }
  OS << "-------------------------------------------------------\n"
     << "Metric 4: Function call inflow <> Net entry(s) outflow\n"
     << "-------------------------------------------------------\n"
     << format(
            "Focus on %zu (%.2lf%%) of considered functions that is called by "
            "at least 1 other function and whose entry BB(s) are not exits\n",
            AbsDiffs.size(),
            100.0 * (double)AbsDiffs.size() /
                (std::distance(Functions.begin(), Functions.end())))
     << "LHS = Function call inflow; RHS = Net entry(s) outflow\n"
     << "MAX = MAX(LHS, RHS); MIN = MIN(LHS, RHS)\n";

  if (!FractionDiffs.empty()) {
    OS << "- Distribution of [(MAX - MIN) / MAX] among all considered "
          "functions\n";
    printDistribution(OS, FractionDiffs, /*Fraction*/ true);
  }
  if (!AbsDiffs.empty()) {
    OS << "- Distribution of [MAX - MIN] among all considered functions\n";
    printDistribution(OS, AbsDiffs);
  }
}

void printCFGFragmentationMetrics(raw_ostream &OS,
                                  iterator_range<function_iterator> &Functions,
                                  FlowInfo &TotalFlowMap) {
  // Given a perfect profile, every positive-execution-count BB should be
  // connected to an entry of the function through a positive-execution-count
  // directed path in the control flow graph.
  std::vector<size_t> NumUnreachables;
  std::vector<size_t> SumECUnreachables;
  std::vector<double> FractionECUnreachables;

  TotalFlowMapTy &TotalIECMaps = TotalFlowMap.TotalIECMaps;

  for (auto it = Functions.begin(); it != Functions.end(); ++it) {
    const BinaryFunction *Function = *it;
    if (Function->size() <= 1)
      continue;

    // Compute a mapping from each BB to its execution count (EC) defiend by
    // MAX(in-flow, out-flow, BB.getKnownExecutionCount()). Compute the sum of
    // all BB ECs.
    FlowMapTy &IECMap = TotalIECMaps[Function];
    size_t NumPosECBBs = 0;
    size_t SumAllBBEC = 0;
    for (const BinaryBasicBlock &BB : *Function) {
      size_t BBEC = std::max(IECMap[&BB], BB.getKnownExecutionCount());
      NumPosECBBs += BBEC > 0 ? 1 : 0;
      SumAllBBEC += BBEC;
    }

    // Perform BFS on subgraph of CFG induced by positive weight jump edges.
    // Compute the number of BBs reachable from the entry(s) of the function and
    // the sum of their ECs.
    std::unordered_map<unsigned, const BinaryBasicBlock *> IndexToBB;
    std::unordered_set<unsigned> Visited;
    std::queue<unsigned> Queue;
    for (const BinaryBasicBlock &BB : *Function) {
      size_t BBEC = std::max(IECMap[&BB], BB.getKnownExecutionCount());
      // Make sure BB.getIndex() is not already in IndexToBB.
      assert(IndexToBB.find(BB.getIndex()) == IndexToBB.end());
      IndexToBB[BB.getIndex()] = &BB;
      if (BB.isEntryPoint() && BBEC > 0) {
        Queue.push(BB.getIndex());
        Visited.insert(BB.getIndex());
      }
    }
    while (!Queue.empty()) {
      unsigned BBIndex = Queue.front();
      const BinaryBasicBlock *BB = IndexToBB[BBIndex];
      Queue.pop();
      auto SuccBIIter = BB->branch_info_begin();
      for (BinaryBasicBlock *Succ : BB->successors()) {
        uint64_t Count = SuccBIIter->Count;
        if (Count == BinaryBasicBlock::COUNT_NO_PROFILE || Count == 0) {
          ++SuccBIIter;
          continue;
        }
        if (!Visited.insert(Succ->getIndex()).second) {
          ++SuccBIIter;
          continue;
        }
        Queue.push(Succ->getIndex());
        ++SuccBIIter;
      }
    }

    size_t NumReachableBBs = Visited.size();

    // Loop through Visited, and sum the corresponding BBs' ECs.
    size_t SumReachableBBEC = 0;
    for (unsigned BBIndex : Visited) {
      const BinaryBasicBlock *BB = IndexToBB[BBIndex];
      size_t BBEC = std::max(IECMap[BB], BB->getKnownExecutionCount());
      SumReachableBBEC += BBEC;
    }

    size_t NumPosECBBsUnreachableFromEntry = NumPosECBBs - NumReachableBBs;
    size_t SumUnreachableBBEC = SumAllBBEC - SumReachableBBEC;
    double FractionECUnreachable = (double)SumUnreachableBBEC / SumAllBBEC;

    if (opts::Verbosity >= 2 && FractionECUnreachable > 0.5 &&
        SumUnreachableBBEC > 500) {
      OS << "Big fragmentation of hot part of CFG observed in function "
         << Function->getPrintName() << "\n";
      LLVM_DEBUG(Function->dump());
    }

    NumUnreachables.push_back(NumPosECBBsUnreachableFromEntry);
    SumECUnreachables.push_back(SumUnreachableBBEC);
    FractionECUnreachables.push_back(FractionECUnreachable);
  }
  OS << "-------------------------------------------------------\n"
     << "Metric 5: Function cfg fragmentation\n"
     << "-------------------------------------------------------\n"
     << format("Focus on %zu (%.2lf%%) of considered functions that have at "
               "least 2 BBs\n",
               NumUnreachables.size(),
               100.0 * (double)NumUnreachables.size() /
                   (std::distance(Functions.begin(), Functions.end())));

  if (!NumUnreachables.empty()) {
    OS << "- Distribution of NUM(unreachable POS BBs) among all focal "
          "functions\n";
    printDistribution(OS, NumUnreachables);
  }
  if (!SumECUnreachables.empty()) {
    OS << "- Distribution of SUM(BBEC of unreachable POS BBs) among all focal "
          "functions\n";
    printDistribution(OS, SumECUnreachables);
  }
  if (!FractionECUnreachables.empty()) {
    OS << "- Distribution of [(SUM(BBEC of unreachable POS BBs) / SUM(all "
          "BBEC))] among all focal functions\n";
    printDistribution(OS, FractionECUnreachables, /*Fraction*/ true);
  }
}

void computeFlowMappings(const BinaryContext &BC, FlowInfo &TotalFlowMap) {
  // Compute jump flow: TotalIncomingMaps, TotalOutgoingMaps.
  TotalFlowMapTy &TotalIncomingMaps = TotalFlowMap.TotalIncomingMaps;
  TotalFlowMapTy &TotalOutgoingMaps = TotalFlowMap.TotalOutgoingMaps;
  for (const auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction *Function = &BFI.second;
    if (Function->empty() || !Function->hasValidProfile())
      continue;
    FlowMapTy &IncomingMap = TotalIncomingMaps[Function];
    FlowMapTy &OutgoingMap = TotalOutgoingMaps[Function];
    for (const BinaryBasicBlock &BB : *Function) {
      uint64_t TotalOutgoing = 0ULL;
      auto SuccBIIter = BB.branch_info_begin();
      for (BinaryBasicBlock *Succ : BB.successors()) {
        const uint64_t Count = SuccBIIter->Count;
        if (Count == BinaryBasicBlock::COUNT_NO_PROFILE || Count == 0) {
          ++SuccBIIter;
          continue;
        }
        TotalOutgoing += Count;
        IncomingMap[Succ] += Count;
        ++SuccBIIter;
      }
      OutgoingMap[&BB] = TotalOutgoing;
    }
  }

  // Compute call flow: TotalIECMaps, CallGraphIncomingMap.
  TotalFlowMapTy &TotalIECMaps = TotalFlowMap.TotalIECMaps;
  TotalFlowMapTy &TotalMaxCallMaps = TotalFlowMap.TotalMaxCallMaps;
  FunctionFlowMapTy &CallGraphIncomingMap = TotalFlowMap.CallGraphIncomingMap;
  for (const auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction *Function = &BFI.second;
    FlowMapTy &IECMap = TotalIECMaps[Function];
    FlowMapTy &MaxCallMap = TotalMaxCallMaps[Function];

    // Update IECMap, MaxCallMap, and CallGraphIncomingMap.
    auto recordCall = [&](const BinaryBasicBlock *SourceBB,
                          const MCSymbol *DestSymbol, const uint64_t Count) {
      if (Count == BinaryBasicBlock::COUNT_NO_PROFILE) {
        if (SourceBB)
          MaxCallMap[SourceBB] = std::max(MaxCallMap[SourceBB], (size_t)0);
        return;
      }
      const BinaryFunction *DstFunc =
          DestSymbol ? BC.getFunctionForSymbol(DestSymbol) : nullptr;
      if (DstFunc)
        CallGraphIncomingMap[DstFunc] += Count;
      if (SourceBB) {
        IECMap[SourceBB] = std::max(IECMap[SourceBB], Count);
        MaxCallMap[SourceBB] = std::max(MaxCallMap[SourceBB], Count);
      }
    };

    // Get pairs of (symbol, count) for each target at this callsite.
    // If the call is to an unknown function the symbol will be nullptr.
    // If there is no profiling data the count will be COUNT_NO_PROFILE.
    using TargetDesc = std::pair<const MCSymbol *, uint64_t>;
    using CallInfoTy = std::vector<TargetDesc>;
    auto getCallInfo = [&](const BinaryBasicBlock *BB, const MCInst &Inst) {
      CallInfoTy Counts;
      const MCSymbol *DstSym = BC.MIB->getTargetSymbol(Inst);

      // If this is an indirect call use perf data directly.
      if (!DstSym && BC.MIB->hasAnnotation(Inst, "CallProfile")) {
        const auto &ICSP = BC.MIB->getAnnotationAs<IndirectCallSiteProfile>(
            Inst, "CallProfile");
        for (const IndirectCallProfile &CSI : ICSP)
          if (CSI.Symbol)
            Counts.emplace_back(CSI.Symbol, CSI.Count);
      } else {
        const uint64_t Count = BB->getExecutionCount();
        Counts.emplace_back(DstSym, Count);
      }

      return Counts;
    };

    // If the function has an invalid profile, try to use the perf data
    // directly. The call EC is only used to update CallGraphIncomingMap.
    if (!Function->hasValidProfile() && !Function->getAllCallSites().empty()) {
      for (const IndirectCallProfile &CSI : Function->getAllCallSites()) {
        if (!CSI.Symbol)
          continue;
        recordCall(nullptr, CSI.Symbol, CSI.Count);
      }
      continue;
    } else {
      // If the function has a valid profile, then use call EC to update
      // both TotalIECMaps and CallGraphIncomingMap.
      FlowMapTy &IncomingMap = TotalIncomingMaps[Function];
      FlowMapTy &OutgoingMap = TotalOutgoingMaps[Function];
      for (BinaryBasicBlock *BB : Function->getLayout().blocks()) {
        IECMap[BB] = std::max(IncomingMap[BB], OutgoingMap[BB]);
        for (MCInst &Inst : *BB) {
          if (!BC.MIB->isCall(Inst))
            continue;
          // Find call instructions and extract target symbols from each one.
          const CallInfoTy CallInfo = getCallInfo(BB, Inst);
          for (const TargetDesc &CI : CallInfo) {
            recordCall(BB, CI.first, CI.second);
          }
        }
      }
    }
  }
}

void printBucketedMetrics(raw_ostream &OS, BinaryContext &BC,
                          size_t NumFunctionsPerBucket = 500,
                          size_t NumTopFunctions = 2500,
                          size_t BBECThreshold = 100) {
  // Create mappings to store BB inflow, outflow, and call EC that are used by
  // the profile quality metrics.
  FlowInfo TotalFlowMap;
  computeFlowMappings(BC, TotalFlowMap);

  // Create a list of functions with valid profiles.
  FunctionListType ValidFunctions;
  for (const auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction *Function = &BFI.second;
    if (Function->empty() || !Function->hasValidProfile() ||
        !Function->isSimple())
      continue;
    ValidFunctions.push_back(Function);
  }

  // Sort the list of functions by execution counts (reverse).
  llvm::sort(ValidFunctions,
             [&](const BinaryFunction *A, const BinaryFunction *B) {
               return A->getKnownExecutionCount() > B->getKnownExecutionCount();
             });

  size_t PerBucketSize = std::min(NumFunctionsPerBucket, ValidFunctions.size());
  if (PerBucketSize == 0)
    return;
  size_t RealNumTopFunctions = std::min(NumTopFunctions, ValidFunctions.size());
  size_t NumBuckets = RealNumTopFunctions / PerBucketSize +
                      (RealNumTopFunctions % PerBucketSize != 0);
  OS << format("BOLT-INFO: Profile quality metrics for the hottest "
               "%zu functions "
               "divided into %zu buckets, each with at most %zu functions:\n",
               RealNumTopFunctions, NumBuckets, PerBucketSize);
  OS << "Abbreviations:\n"
     << "* EC     : execution count\n"
     << "* BB     : binary basic block\n"
     << "* KEC    : BinaryBasicBlock::getKnownExecutionCount()\n"
     << "* IEC    : Inferred Basic Block EC := MAX(BB SUM inflow, BB SUM "
        "outflow, BB "
        "individual call ECs including indirect calls)\n";
  // Create NumBuckets iterator_ranges, one for each bucket
  for (size_t BucketIndex = 0; BucketIndex < NumBuckets; ++BucketIndex) {
    const size_t StartIndex = BucketIndex * PerBucketSize;
    size_t EndIndex = std::min(StartIndex + PerBucketSize, NumTopFunctions);
    EndIndex = std::min(EndIndex, ValidFunctions.size());
    iterator_range<function_iterator> Functions(
        ValidFunctions.begin() + StartIndex, ValidFunctions.begin() + EndIndex);
    const size_t MaxFunctionExecutionCount =
        ValidFunctions[StartIndex]->getKnownExecutionCount();
    const size_t MinFunctionExecutionCount =
        ValidFunctions[EndIndex - 1]->getKnownExecutionCount();
    size_t NumBBs = 0;
    size_t NumBBsECThreshold = 0;
    for (const BinaryFunction *Function : Functions) {
      FlowMapTy &IECMap = TotalFlowMap.TotalIECMaps[Function];
      for (const BinaryBasicBlock &BB : *Function) {
        NumBBs++;
        if (std::max(IECMap[&BB], BB.getKnownExecutionCount()) >= BBECThreshold)
          NumBBsECThreshold++;
      }
    }
    OS << format("\n----------------\n|   Bucket %zu:  "
                 "|\n----------------\nECs of the %zu functions in the bucket: "
                 "%zu-%zu\nConsidering %zu BBs (out of %zu /  %.2lf%% of all "
                 "BBs in the bucket) that have BBEC := MAX(KEC, IEC) >= %zu\n",
                 BucketIndex + 1, EndIndex - StartIndex,
                 MinFunctionExecutionCount, MaxFunctionExecutionCount,
                 NumBBsECThreshold, NumBBs,
                 100.0 * (double)NumBBsECThreshold / NumBBs, BBECThreshold);
    printKECMetrics(OS, Functions, BBECThreshold, TotalFlowMap);
    printFlowConservationMetrics(OS, Functions, BBECThreshold, TotalFlowMap);
    printFlowCallConservationMetrics(OS, Functions, BBECThreshold,
                                     TotalFlowMap);
    printCallGraphConsistencyMetrics(OS, Functions, TotalFlowMap);
    printCFGFragmentationMetrics(OS, Functions, TotalFlowMap);
  }
}
} // namespace

void ProfileStats::printAll(raw_ostream &OS, BinaryContext &BC) {
  printProfileBiasScore(OS, BC);
  if (opts::PrintBucketedMetrics)
    printBucketedMetrics(OS, BC, opts::NumFunctionsPerBucket,
                         opts::NumTopFunctions, opts::BBECThreshold);
}

//===- bolt/Passes/ProfileQualityStats.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the profile quality stats calculation pass.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/ProfileQualityStats.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/Support/CommandLine.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>

using namespace llvm;
using namespace bolt;

namespace opts {
extern cl::opt<unsigned> Verbosity;
cl::opt<unsigned> NumFunctionsForProfileQualityCheck(
    "num-functions-for-profile-quality-check",
    cl::desc("number of hottest functions to print aggregated "
             "profile quality stats of."),
    cl::init(1000), cl::ZeroOrMore, cl::Hidden, cl::cat(BoltOptCategory));
cl::opt<unsigned> PercentileForProfileQualityCheck(
    "percentile-for-profile-quality-check",
    cl::desc("Percentile of profile quality distributions over hottest "
             "functions to report."),
    cl::init(95), cl::ZeroOrMore, cl::Hidden, cl::cat(BoltOptCategory));
} // namespace opts

namespace {
using FunctionListType = std::vector<const BinaryFunction *>;
using function_iterator = FunctionListType::iterator;

// BB index -> flow count
using FlowMapTy = std::unordered_map<unsigned, uint64_t>;
// Function number -> FlowMapTy
using TotalFlowMapTy = std::unordered_map<uint64_t, FlowMapTy>;
// Function number -> flow count
using FunctionFlowMapTy = std::unordered_map<uint64_t, uint64_t>;
struct FlowInfo {
  TotalFlowMapTy TotalIncomingMaps;
  TotalFlowMapTy TotalOutgoingMaps;
  TotalFlowMapTy TotalMaxCountMaps;
  TotalFlowMapTy TotalMinCountMaps;
  FunctionFlowMapTy CallGraphIncomingMap;
};

template <typename T>
void printDistribution(raw_ostream &OS, std::vector<T> &values,
                       bool Fraction = false) {
  // Assume values are sorted.
  if (values.empty())
    return;

  OS << "  Length     : " << values.size() << "\n";

  auto printLine = [&](std::string Text, double Percent) {
    int Rank = int(values.size() * (100 - Percent) / 100);
    if (Percent == 0)
      Rank = values.size() - 1;
    if (Fraction)
      OS << "  " << Text << std::string(11 - Text.length(), ' ') << ": "
         << format("%.2lf%%", values[Rank] * 100) << "\n";
    else
      OS << "  " << Text << std::string(11 - Text.length(), ' ') << ": "
         << values[Rank] << "\n";
  };

  printLine("MAX", 0);
  const int percentages[] = {1, 5, 10, 20, 50, 80};
  for (size_t i = 0; i < sizeof(percentages) / sizeof(percentages[0]); ++i) {
    printLine("TOP " + std::to_string(percentages[i]) + "%", percentages[i]);
  }
  printLine("MIN", 100);
}

void printCFGContinuityStats(raw_ostream &OS,
                             iterator_range<function_iterator> &Functions) {
  // Given a perfect profile, every positive-execution-count BB should be
  // connected to an entry of the function through a positive-execution-count
  // directed path in the control flow graph.
  std::vector<size_t> NumUnreachables;
  std::vector<size_t> SumECUnreachables;
  std::vector<double> FractionECUnreachables;

  for (const BinaryFunction *Function : Functions) {
    if (Function->size() <= 1)
      continue;

    // Compute the sum of all BB execution counts (ECs).
    size_t NumPosECBBs = 0;
    size_t SumAllBBEC = 0;
    for (const BinaryBasicBlock &BB : *Function) {
      const size_t BBEC = BB.getKnownExecutionCount();
      NumPosECBBs += !!BBEC;
      SumAllBBEC += BBEC;
    }

    // Perform BFS on subgraph of CFG induced by positive weight edges.
    // Compute the number of BBs reachable from the entry(s) of the function and
    // the sum of their execution counts (ECs).
    std::unordered_set<unsigned> Visited;
    std::queue<unsigned> Queue;
    size_t SumReachableBBEC = 0;

    Function->forEachEntryPoint([&](uint64_t Offset, const MCSymbol *Label) {
      const BinaryBasicBlock *EntryBB = Function->getBasicBlockAtOffset(Offset);
      if (EntryBB && EntryBB->getKnownExecutionCount() > 0) {
        Queue.push(EntryBB->getLayoutIndex());
        Visited.insert(EntryBB->getLayoutIndex());
        SumReachableBBEC += EntryBB->getKnownExecutionCount();
      }
      return true;
    });

    const FunctionLayout &Layout = Function->getLayout();

    while (!Queue.empty()) {
      const unsigned BBIndex = Queue.front();
      const BinaryBasicBlock *BB = Layout.getBlock(BBIndex);
      Queue.pop();
      auto SuccBIIter = BB->branch_info_begin();
      for (const BinaryBasicBlock *Succ : BB->successors()) {
        const uint64_t Count = SuccBIIter->Count;
        if (Count == BinaryBasicBlock::COUNT_NO_PROFILE || Count == 0) {
          ++SuccBIIter;
          continue;
        }
        if (!Visited.insert(Succ->getLayoutIndex()).second) {
          ++SuccBIIter;
          continue;
        }
        SumReachableBBEC += Succ->getKnownExecutionCount();
        Queue.push(Succ->getLayoutIndex());
        ++SuccBIIter;
      }
    }

    const size_t NumReachableBBs = Visited.size();

    const size_t NumPosECBBsUnreachableFromEntry =
        NumPosECBBs - NumReachableBBs;
    const size_t SumUnreachableBBEC = SumAllBBEC - SumReachableBBEC;
    const double FractionECUnreachable =
        (double)SumUnreachableBBEC / SumAllBBEC;

    if (opts::Verbosity >= 2 && FractionECUnreachable >= 0.05) {
      OS << "Non-trivial CFG discontinuity observed in function "
         << Function->getPrintName() << "\n";
      if (opts::Verbosity >= 3)
        Function->dump();
    }

    NumUnreachables.push_back(NumPosECBBsUnreachableFromEntry);
    SumECUnreachables.push_back(SumUnreachableBBEC);
    FractionECUnreachables.push_back(FractionECUnreachable);
  }

  if (FractionECUnreachables.empty())
    return;

  std::sort(FractionECUnreachables.begin(), FractionECUnreachables.end());
  const int Rank = int(FractionECUnreachables.size() *
                       opts::PercentileForProfileQualityCheck / 100);
  OS << format("function CFG discontinuity %.2lf%%; ",
               FractionECUnreachables[Rank] * 100);
  if (opts::Verbosity >= 1) {
    OS << "\nabbreviations: EC = execution count, POS BBs = positive EC BBs\n"
       << "distribution of NUM(unreachable POS BBs) per function\n";
    std::sort(NumUnreachables.begin(), NumUnreachables.end());
    printDistribution(OS, NumUnreachables);

    OS << "distribution of SUM_EC(unreachable POS BBs) per function\n";
    std::sort(SumECUnreachables.begin(), SumECUnreachables.end());
    printDistribution(OS, SumECUnreachables);

    OS << "distribution of [(SUM_EC(unreachable POS BBs) / SUM_EC(all "
          "POS BBs))] per function\n";
    printDistribution(OS, FractionECUnreachables, /*Fraction=*/true);
  }
}

void printCallGraphFlowConservationStats(
    raw_ostream &OS, iterator_range<function_iterator> &Functions,
    FlowInfo &TotalFlowMap) {
  std::vector<double> CallGraphGaps;

  for (const BinaryFunction *Function : Functions) {
    if (Function->size() <= 1 || !Function->isSimple())
      continue;

    const uint64_t FunctionNum = Function->getFunctionNumber();
    FlowMapTy &IncomingMap = TotalFlowMap.TotalIncomingMaps[FunctionNum];
    FlowMapTy &OutgoingMap = TotalFlowMap.TotalOutgoingMaps[FunctionNum];
    FunctionFlowMapTy &CallGraphIncomingMap = TotalFlowMap.CallGraphIncomingMap;

    // Only consider functions that are not a program entry.
    if (CallGraphIncomingMap.find(FunctionNum) != CallGraphIncomingMap.end()) {
      uint64_t EntryInflow = 0;
      uint64_t EntryOutflow = 0;
      uint32_t NumConsideredEntryBlocks = 0;
      for (const BinaryBasicBlock &BB : *Function) {
        if (BB.isEntryPoint()) {
          // If entry is an exit, then we don't consider it for flow
          // conservation
          if (BB.succ_size() == 0)
            continue;
          NumConsideredEntryBlocks++;

          EntryInflow += IncomingMap[BB.getLayoutIndex()];
          EntryOutflow += OutgoingMap[BB.getLayoutIndex()];
        }
      }
      uint64_t NetEntryOutflow = 0;
      if (EntryOutflow < EntryInflow) {
        if (opts::Verbosity >= 2) {
          // We expect entry blocks' CFG outflow >= inflow, i.e., it has a
          // non-negative net outflow. If this is not the case, then raise a
          // warning if requested.
          OS << "BOLT WARNING: unexpected entry block CFG outflow < inflow "
                "in "
                "function "
             << Function->getPrintName() << "\n";
          if (opts::Verbosity >= 3)
            Function->dump();
        }
      } else {
        NetEntryOutflow = EntryOutflow - EntryInflow;
      }
      if (NumConsideredEntryBlocks > 0) {
        const uint64_t CallGraphInflow =
            TotalFlowMap.CallGraphIncomingMap[Function->getFunctionNumber()];
        const uint64_t Min = std::min(NetEntryOutflow, CallGraphInflow);
        const uint64_t Max = std::max(NetEntryOutflow, CallGraphInflow);
        const double CallGraphGap = 1 - (double)Min / Max;

        if (opts::Verbosity >= 2 && CallGraphGap >= 0.5) {
          OS << "Nontrivial call graph gap of size "
             << format("%.2lf%%", 100 * CallGraphGap)
             << " observed in function " << Function->getPrintName() << "\n";
          if (opts::Verbosity >= 3)
            Function->dump();
        }

        CallGraphGaps.push_back(CallGraphGap);
      }
    }
  }

  if (CallGraphGaps.empty())
    return;

  std::sort(CallGraphGaps.begin(), CallGraphGaps.end());
  const int Rank =
      int(CallGraphGaps.size() * opts::PercentileForProfileQualityCheck / 100);
  OS << format("call graph flow conservation gap %.2lf%%; ",
               CallGraphGaps[Rank] * 100);
  if (opts::Verbosity >= 1) {
    OS << "\ndistribution of function entry flow conservation gaps\n";
    printDistribution(OS, CallGraphGaps, /*Fraction=*/true);
  }
}

void printCFGFlowConservationStats(raw_ostream &OS,
                                   iterator_range<function_iterator> &Functions,
                                   FlowInfo &TotalFlowMap) {
  std::vector<double> CFGGapsWeightedAvg;
  std::vector<double> CFGGapsWorst;
  std::vector<uint64_t> CFGGapsWorstAbs;

  for (const BinaryFunction *Function : Functions) {
    if (Function->size() <= 1 || !Function->isSimple())
      continue;

    const uint64_t FunctionNum = Function->getFunctionNumber();
    FlowMapTy &MaxCountMaps = TotalFlowMap.TotalMaxCountMaps[FunctionNum];
    FlowMapTy &MinCountMaps = TotalFlowMap.TotalMinCountMaps[FunctionNum];
    double WeightedGapSum = 0.0;
    double WeightSum = 0.0;
    double WorstGap = 0.0;
    uint64_t WorstGapAbs = 0;
    BinaryBasicBlock *BBWorstGap = nullptr;
    BinaryBasicBlock *BBWorstGapAbs = nullptr;
    for (BinaryBasicBlock &BB : *Function) {
      // We don't consider function entry or exit blocks for CFG flow
      // conservation
      if (BB.isEntryPoint() || BB.succ_size() == 0)
        continue;

      const uint64_t Max = MaxCountMaps[BB.getLayoutIndex()];
      const uint64_t Min = MinCountMaps[BB.getLayoutIndex()];
      const double Gap = 1 - (double)Min / Max;
      double Weight = BB.getKnownExecutionCount() * BB.getNumNonPseudos();
      if (Weight == 0)
        continue;
      // We use log to prevent the stats from being dominated by extremely hot
      // blocks
      Weight = log(Weight);
      WeightedGapSum += Gap * Weight;
      WeightSum += Weight;
      if (BB.getKnownExecutionCount() > 500 && Gap > WorstGap) {
        WorstGap = Gap;
        BBWorstGap = &BB;
      }
      if (BB.getKnownExecutionCount() > 500 && Max - Min > WorstGapAbs) {
        WorstGapAbs = Max - Min;
        BBWorstGapAbs = &BB;
      }
    }
    if (WeightSum > 0) {
      const double WeightedGap = WeightedGapSum / WeightSum;
      if (opts::Verbosity >= 2 && (WeightedGap >= 0.1 || WorstGap >= 0.9)) {
        OS << "Nontrivial CFG gap observed in function "
           << Function->getPrintName() << "\n"
           << "Weighted gap: " << format("%.2lf%%", 100 * WeightedGap) << "\n";
        if (BBWorstGap)
          OS << "Worst gap: " << format("%.2lf%%", 100 * WorstGap)
             << " at BB with input offset: 0x"
             << Twine::utohexstr(BBWorstGap->getInputOffset()) << "\n";
        if (BBWorstGapAbs)
          OS << "Worst gap (absolute value): " << WorstGapAbs << " at BB with "
             << "input offset 0x"
             << Twine::utohexstr(BBWorstGapAbs->getInputOffset()) << "\n";
        if (opts::Verbosity >= 3)
          Function->dump();
      }

      CFGGapsWeightedAvg.push_back(WeightedGap);
      CFGGapsWorst.push_back(WorstGap);
      CFGGapsWorstAbs.push_back(WorstGapAbs);
    }
  }

  if (CFGGapsWeightedAvg.empty())
    return;
  std::sort(CFGGapsWeightedAvg.begin(), CFGGapsWeightedAvg.end());
  const int RankWA = int(CFGGapsWeightedAvg.size() *
                         opts::PercentileForProfileQualityCheck / 100);
  std::sort(CFGGapsWorst.begin(), CFGGapsWorst.end());
  const int RankW =
      int(CFGGapsWorst.size() * opts::PercentileForProfileQualityCheck / 100);
  OS << format("CFG flow conservation gap %.2lf%% (weighted) %.2lf%% (worst)\n",
               CFGGapsWeightedAvg[RankWA] * 100, CFGGapsWorst[RankW] * 100);
  if (opts::Verbosity >= 1) {
    OS << "distribution of weighted CFG flow conservation gaps\n";
    printDistribution(OS, CFGGapsWeightedAvg, /*Fraction=*/true);
    OS << "Consider only blocks with execution counts > 500:\n"
       << "distribution of worst block flow conservation gap per "
          "function \n";
    printDistribution(OS, CFGGapsWorst, /*Fraction=*/true);
    OS << "distribution of worst block flow conservation gap (absolute "
          "value) per function\n";
    std::sort(CFGGapsWorstAbs.begin(), CFGGapsWorstAbs.end());
    printDistribution(OS, CFGGapsWorstAbs, /*Fraction=*/false);
  }
}

void computeFlowMappings(const BinaryContext &BC, FlowInfo &TotalFlowMap) {
  // Increment block inflow and outflow with CFG jump counts.
  TotalFlowMapTy &TotalIncomingMaps = TotalFlowMap.TotalIncomingMaps;
  TotalFlowMapTy &TotalOutgoingMaps = TotalFlowMap.TotalOutgoingMaps;
  for (const auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction *Function = &BFI.second;
    if (Function->empty() || !Function->hasValidProfile())
      continue;
    FlowMapTy &IncomingMap = TotalIncomingMaps[Function->getFunctionNumber()];
    FlowMapTy &OutgoingMap = TotalOutgoingMaps[Function->getFunctionNumber()];
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
        IncomingMap[Succ->getLayoutIndex()] += Count;
        ++SuccBIIter;
      }
      OutgoingMap[BB.getLayoutIndex()] = TotalOutgoing;
    }
  }

  // Initialize TotalMaxCountMaps and TotalMinCountMaps using
  // TotalIncomingMaps and TotalOutgoingMaps
  TotalFlowMapTy &TotalMaxCountMaps = TotalFlowMap.TotalMaxCountMaps;
  TotalFlowMapTy &TotalMinCountMaps = TotalFlowMap.TotalMinCountMaps;
  for (const auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction *Function = &BFI.second;
    if (Function->empty() || !Function->hasValidProfile())
      continue;
    uint64_t FunctionNum = Function->getFunctionNumber();
    FlowMapTy &IncomingMap = TotalIncomingMaps[FunctionNum];
    FlowMapTy &OutgoingMap = TotalOutgoingMaps[FunctionNum];
    FlowMapTy &MaxCountMap = TotalMaxCountMaps[FunctionNum];
    FlowMapTy &MinCountMap = TotalMinCountMaps[FunctionNum];
    for (const BinaryBasicBlock &BB : *Function) {
      uint64_t BBNum = BB.getLayoutIndex();
      MaxCountMap[BBNum] = std::max(IncomingMap[BBNum], OutgoingMap[BBNum]);
      MinCountMap[BBNum] = std::min(IncomingMap[BBNum], OutgoingMap[BBNum]);
    }
  }

  // Modify TotalMaxCountMaps and TotalMinCountMaps using call counts and
  // fill out CallGraphIncomingMap
  FunctionFlowMapTy &CallGraphIncomingMap = TotalFlowMap.CallGraphIncomingMap;
  for (const auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction *Function = &BFI.second;
    uint64_t FunctionNum = Function->getFunctionNumber();
    FlowMapTy &MaxCountMap = TotalMaxCountMaps[FunctionNum];
    FlowMapTy &MinCountMap = TotalMinCountMaps[FunctionNum];

    // Update MaxCountMap, MinCountMap, and CallGraphIncomingMap
    auto recordCall = [&](const BinaryBasicBlock *SourceBB,
                          const MCSymbol *DestSymbol, uint64_t Count,
                          uint64_t TotalCallCount) {
      if (Count == BinaryBasicBlock::COUNT_NO_PROFILE)
        Count = 0;
      const BinaryFunction *DstFunc =
          DestSymbol ? BC.getFunctionForSymbol(DestSymbol) : nullptr;
      if (DstFunc)
        CallGraphIncomingMap[DstFunc->getFunctionNumber()] += Count;
      if (SourceBB) {
        unsigned BlockIndex = SourceBB->getLayoutIndex();
        MaxCountMap[BlockIndex] =
            std::max(MaxCountMap[BlockIndex], TotalCallCount);
        MinCountMap[BlockIndex] =
            std::min(MinCountMap[BlockIndex], TotalCallCount);
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
        if (CSI.Symbol)
          recordCall(nullptr, CSI.Symbol, CSI.Count, CSI.Count);
      }
      continue;
    } else {
      // If the function has a valid profile
      for (const BinaryBasicBlock &BB : *Function) {
        for (const MCInst &Inst : BB) {
          if (BC.MIB->isCall(Inst)) {
            // Find call instructions and extract target symbols from each
            // one.
            const CallInfoTy CallInfo = getCallInfo(&BB, Inst);
            // We need the total call count to update MaxCountMap and
            // MinCountMap in recordCall for indirect calls
            uint64_t TotalCallCount = 0;
            for (const TargetDesc &CI : CallInfo)
              TotalCallCount += CI.second;
            for (const TargetDesc &CI : CallInfo)
              recordCall(&BB, CI.first, CI.second, TotalCallCount);
          }
        }
      }
    }
  }
}

void printAll(BinaryContext &BC, FunctionListType &ValidFunctions,
              size_t NumTopFunctions) {
  // Sort the list of functions by execution counts (reverse).
  llvm::sort(ValidFunctions,
             [&](const BinaryFunction *A, const BinaryFunction *B) {
               return A->getKnownExecutionCount() > B->getKnownExecutionCount();
             });

  const size_t RealNumTopFunctions =
      std::min(NumTopFunctions, ValidFunctions.size());

  iterator_range<function_iterator> Functions(
      ValidFunctions.begin(), ValidFunctions.begin() + RealNumTopFunctions);

  FlowInfo TotalFlowMap;
  computeFlowMappings(BC, TotalFlowMap);

  BC.outs() << format("BOLT-INFO: profile quality metrics for the hottest %zu "
                      "functions (reporting top %zu%% values): ",
                      RealNumTopFunctions,
                      100 - opts::PercentileForProfileQualityCheck);
  printCFGContinuityStats(BC.outs(), Functions);
  printCallGraphFlowConservationStats(BC.outs(), Functions, TotalFlowMap);
  printCFGFlowConservationStats(BC.outs(), Functions, TotalFlowMap);

  // Print more detailed bucketed stats if requested.
  if (opts::Verbosity >= 1 && RealNumTopFunctions >= 5) {
    const size_t PerBucketSize = RealNumTopFunctions / 5;
    BC.outs() << format(
        "Detailed stats for 5 buckets, each with  %zu functions:\n",
        PerBucketSize);

    // For each bucket, print the CFG continuity stats of the functions in
    // the bucket.
    for (size_t BucketIndex = 0; BucketIndex < 5; ++BucketIndex) {
      const size_t StartIndex = BucketIndex * PerBucketSize;
      const size_t EndIndex = StartIndex + PerBucketSize;
      iterator_range<function_iterator> Functions(
          ValidFunctions.begin() + StartIndex,
          ValidFunctions.begin() + EndIndex);
      const size_t MaxFunctionExecutionCount =
          ValidFunctions[StartIndex]->getKnownExecutionCount();
      const size_t MinFunctionExecutionCount =
          ValidFunctions[EndIndex - 1]->getKnownExecutionCount();
      BC.outs() << format("----------------\n|   Bucket %zu:  "
                          "|\n----------------\n",
                          BucketIndex + 1)
                << format(
                       "execution counts of the %zu functions in the bucket: "
                       "%zu-%zu\n",
                       EndIndex - StartIndex, MinFunctionExecutionCount,
                       MaxFunctionExecutionCount);
      printCFGContinuityStats(BC.outs(), Functions);
      printCallGraphFlowConservationStats(BC.outs(), Functions, TotalFlowMap);
      printCFGFlowConservationStats(BC.outs(), Functions, TotalFlowMap);
    }
  }
}
} // namespace

bool PrintProfileQualityStats::shouldOptimize(const BinaryFunction &BF) const {
  if (BF.empty() || !BF.hasValidProfile())
    return false;

  return BinaryFunctionPass::shouldOptimize(BF);
}

Error PrintProfileQualityStats::runOnFunctions(BinaryContext &BC) {
  // Create a list of functions with valid profiles.
  FunctionListType ValidFunctions;
  for (const auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction *Function = &BFI.second;
    if (PrintProfileQualityStats::shouldOptimize(*Function))
      ValidFunctions.push_back(Function);
  }
  if (ValidFunctions.empty() || opts::NumFunctionsForProfileQualityCheck == 0)
    return Error::success();

  printAll(BC, ValidFunctions, opts::NumFunctionsForProfileQualityCheck);
  return Error::success();
}

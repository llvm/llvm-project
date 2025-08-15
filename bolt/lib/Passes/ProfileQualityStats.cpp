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
static cl::opt<unsigned> TopFunctionsForProfileQualityCheck(
    "top-functions-for-profile-quality-check",
    cl::desc("number of hottest functions to print aggregated "
             "profile quality stats of."),
    cl::init(1000), cl::ZeroOrMore, cl::Hidden, cl::cat(BoltOptCategory));
static cl::opt<unsigned> PercentileForProfileQualityCheck(
    "percentile-for-profile-quality-check",
    cl::desc("Percentile of profile quality distributions over hottest "
             "functions to report."),
    cl::init(95), cl::ZeroOrMore, cl::Hidden, cl::cat(BoltOptCategory));
} // namespace opts

namespace {
using FunctionListType = std::vector<const BinaryFunction *>;
using function_iterator = FunctionListType::iterator;

// Function number -> vector of flows for BBs in the function
using TotalFlowMapTy = std::unordered_map<uint64_t, std::vector<uint64_t>>;
// Function number -> flow count
using FunctionFlowMapTy = std::unordered_map<uint64_t, uint64_t>;
struct FlowInfo {
  TotalFlowMapTy TotalIncomingFlows;
  TotalFlowMapTy TotalOutgoingFlows;
  TotalFlowMapTy TotalMaxCountMaps;
  TotalFlowMapTy TotalMinCountMaps;
  FunctionFlowMapTy CallGraphIncomingFlows;
};

// When reporting exception handling stats, we only consider functions with at
// least MinLPECSum counts in landing pads to avoid false positives due to
// sampling noise
const uint16_t MinLPECSum = 50;

// When reporting CFG flow conservation stats, we only consider blocks with
// execution counts > MinBlockCount when reporting the distribution of worst
// gaps.
const uint16_t MinBlockCount = 500;

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
         << formatv("{0:P}", values[Rank]) << "\n";
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
    if (Function->size() <= 1) {
      NumUnreachables.push_back(0);
      SumECUnreachables.push_back(0);
      FractionECUnreachables.push_back(0.0);
      continue;
    }

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
      if (!EntryBB || EntryBB->getKnownExecutionCount() == 0)
        return true;
      Queue.push(EntryBB->getLayoutIndex());
      Visited.insert(EntryBB->getLayoutIndex());
      SumReachableBBEC += EntryBB->getKnownExecutionCount();
      return true;
    });

    const FunctionLayout &Layout = Function->getLayout();

    while (!Queue.empty()) {
      const unsigned BBIndex = Queue.front();
      const BinaryBasicBlock *BB = Layout.getBlock(BBIndex);
      Queue.pop();
      for (const auto &[Succ, BI] :
           llvm::zip(BB->successors(), BB->branch_info())) {
        const uint64_t Count = BI.Count;
        if (Count == BinaryBasicBlock::COUNT_NO_PROFILE || Count == 0 ||
            !Visited.insert(Succ->getLayoutIndex()).second)
          continue;
        SumReachableBBEC += Succ->getKnownExecutionCount();
        Queue.push(Succ->getLayoutIndex());
      }
    }

    const size_t NumReachableBBs = Visited.size();

    const size_t NumPosECBBsUnreachableFromEntry =
        NumPosECBBs - NumReachableBBs;
    const size_t SumUnreachableBBEC = SumAllBBEC - SumReachableBBEC;

    double FractionECUnreachable = 0.0;
    if (SumAllBBEC > 0)
      FractionECUnreachable = (double)SumUnreachableBBEC / SumAllBBEC;

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

  llvm::sort(FractionECUnreachables);
  const int Rank = int(FractionECUnreachables.size() *
                       opts::PercentileForProfileQualityCheck / 100);
  OS << formatv("function CFG discontinuity {0:P}; ",
                FractionECUnreachables[Rank]);
  if (opts::Verbosity >= 1) {
    OS << "\nabbreviations: EC = execution count, POS BBs = positive EC BBs\n"
       << "distribution of NUM(unreachable POS BBs) per function\n";
    llvm::sort(NumUnreachables);
    printDistribution(OS, NumUnreachables);

    OS << "distribution of SUM_EC(unreachable POS BBs) per function\n";
    llvm::sort(SumECUnreachables);
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
    if (Function->size() <= 1 || !Function->isSimple()) {
      CallGraphGaps.push_back(0.0);
      continue;
    }

    const uint64_t FunctionNum = Function->getFunctionNumber();
    std::vector<uint64_t> &IncomingFlows =
        TotalFlowMap.TotalIncomingFlows[FunctionNum];
    std::vector<uint64_t> &OutgoingFlows =
        TotalFlowMap.TotalOutgoingFlows[FunctionNum];
    FunctionFlowMapTy &CallGraphIncomingFlows =
        TotalFlowMap.CallGraphIncomingFlows;

    // Only consider functions that are not a program entry.
    if (CallGraphIncomingFlows.find(FunctionNum) ==
        CallGraphIncomingFlows.end()) {
      CallGraphGaps.push_back(0.0);
      continue;
    }

    uint64_t EntryInflow = 0;
    uint64_t EntryOutflow = 0;
    uint32_t NumConsideredEntryBlocks = 0;

    Function->forEachEntryPoint([&](uint64_t Offset, const MCSymbol *Label) {
      const BinaryBasicBlock *EntryBB = Function->getBasicBlockAtOffset(Offset);
      if (!EntryBB || EntryBB->succ_size() == 0)
        return true;
      NumConsideredEntryBlocks++;
      EntryInflow += IncomingFlows[EntryBB->getLayoutIndex()];
      EntryOutflow += OutgoingFlows[EntryBB->getLayoutIndex()];
      return true;
    });

    uint64_t NetEntryOutflow = 0;
    if (EntryOutflow < EntryInflow) {
      if (opts::Verbosity >= 2) {
        // We expect entry blocks' CFG outflow >= inflow, i.e., it has a
        // non-negative net outflow. If this is not the case, then raise a
        // warning if requested.
        OS << "BOLT WARNING: unexpected entry block CFG outflow < inflow "
              "in function "
           << Function->getPrintName() << "\n";
        if (opts::Verbosity >= 3)
          Function->dump();
      }
    } else {
      NetEntryOutflow = EntryOutflow - EntryInflow;
    }
    if (NumConsideredEntryBlocks > 0) {
      const uint64_t CallGraphInflow =
          TotalFlowMap.CallGraphIncomingFlows[Function->getFunctionNumber()];
      const uint64_t Min = std::min(NetEntryOutflow, CallGraphInflow);
      const uint64_t Max = std::max(NetEntryOutflow, CallGraphInflow);
      double CallGraphGap = 0.0;
      if (Max > 0)
        CallGraphGap = 1 - (double)Min / Max;

      if (opts::Verbosity >= 2 && CallGraphGap >= 0.5) {
        OS << "Non-trivial call graph gap of size "
           << formatv("{0:P}", CallGraphGap) << " observed in function "
           << Function->getPrintName() << "\n";
        if (opts::Verbosity >= 3)
          Function->dump();
      }

      CallGraphGaps.push_back(CallGraphGap);
    } else {
      CallGraphGaps.push_back(0.0);
    }
  }

  llvm::sort(CallGraphGaps);
  const int Rank =
      int(CallGraphGaps.size() * opts::PercentileForProfileQualityCheck / 100);
  OS << formatv("call graph flow conservation gap {0:P}; ",
                CallGraphGaps[Rank]);
  if (opts::Verbosity >= 1) {
    OS << "\ndistribution of function entry flow conservation gaps\n";
    printDistribution(OS, CallGraphGaps, /*Fraction=*/true);
  }
}

void printCFGFlowConservationStats(const BinaryContext &BC, raw_ostream &OS,
                                   iterator_range<function_iterator> &Functions,
                                   FlowInfo &TotalFlowMap) {
  std::vector<double> CFGGapsWeightedAvg;
  std::vector<double> CFGGapsWorst;
  std::vector<uint64_t> CFGGapsWorstAbs;
  for (const BinaryFunction *Function : Functions) {
    if (Function->size() <= 1 || !Function->isSimple()) {
      CFGGapsWeightedAvg.push_back(0.0);
      CFGGapsWorst.push_back(0.0);
      CFGGapsWorstAbs.push_back(0);
      continue;
    }

    const uint64_t FunctionNum = Function->getFunctionNumber();
    std::vector<uint64_t> &MaxCountMaps =
        TotalFlowMap.TotalMaxCountMaps[FunctionNum];
    std::vector<uint64_t> &MinCountMaps =
        TotalFlowMap.TotalMinCountMaps[FunctionNum];
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

      if (BB.getKnownExecutionCount() == 0 || BB.getNumNonPseudos() == 0)
        continue;

      // We don't consider blocks that is a landing pad or has a
      // positive-execution-count landing pad
      if (BB.isLandingPad())
        continue;

      if (llvm::any_of(BB.landing_pads(),
                       std::mem_fn(&BinaryBasicBlock::getKnownExecutionCount)))
        continue;

      // We don't consider blocks that end with a recursive call instruction
      const MCInst *Inst = BB.getLastNonPseudoInstr();
      if (BC.MIB->isCall(*Inst)) {
        const MCSymbol *DstSym = BC.MIB->getTargetSymbol(*Inst);
        const BinaryFunction *DstFunc =
            DstSym ? BC.getFunctionForSymbol(DstSym) : nullptr;
        if (DstFunc == Function)
          continue;
      }

      const uint64_t Max = MaxCountMaps[BB.getLayoutIndex()];
      const uint64_t Min = MinCountMaps[BB.getLayoutIndex()];
      double Gap = 0.0;
      if (Max > 0)
        Gap = 1 - (double)Min / Max;
      double Weight = BB.getKnownExecutionCount() * BB.getNumNonPseudos();
      // We use log to prevent the stats from being dominated by extremely hot
      // blocks
      Weight = log(Weight);
      WeightedGapSum += Gap * Weight;
      WeightSum += Weight;
      if (BB.getKnownExecutionCount() > MinBlockCount && Gap > WorstGap) {
        WorstGap = Gap;
        BBWorstGap = &BB;
      }
      if (BB.getKnownExecutionCount() > MinBlockCount &&
          Max - Min > WorstGapAbs) {
        WorstGapAbs = Max - Min;
        BBWorstGapAbs = &BB;
      }
    }
    double WeightedGap = WeightedGapSum;
    if (WeightSum > 0)
      WeightedGap /= WeightSum;
    if (opts::Verbosity >= 2 && WorstGap >= 0.9) {
      OS << "Non-trivial CFG gap observed in function "
         << Function->getPrintName() << "\n"
         << "Weighted gap: " << formatv("{0:P}", WeightedGap) << "\n";
      if (BBWorstGap)
        OS << "Worst gap: " << formatv("{0:P}", WorstGap)
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

  llvm::sort(CFGGapsWeightedAvg);
  const int RankWA = int(CFGGapsWeightedAvg.size() *
                         opts::PercentileForProfileQualityCheck / 100);
  llvm::sort(CFGGapsWorst);
  const int RankW =
      int(CFGGapsWorst.size() * opts::PercentileForProfileQualityCheck / 100);
  OS << formatv("CFG flow conservation gap {0:P} (weighted) {1:P} (worst); ",
                CFGGapsWeightedAvg[RankWA], CFGGapsWorst[RankW]);
  if (opts::Verbosity >= 1) {
    OS << "distribution of weighted CFG flow conservation gaps\n";
    printDistribution(OS, CFGGapsWeightedAvg, /*Fraction=*/true);
    OS << format("Consider only blocks with execution counts > %zu:\n",
                 MinBlockCount)
       << "distribution of worst block flow conservation gap per "
          "function \n";
    printDistribution(OS, CFGGapsWorst, /*Fraction=*/true);
    OS << "distribution of worst block flow conservation gap (absolute "
          "value) per function\n";
    llvm::sort(CFGGapsWorstAbs);
    printDistribution(OS, CFGGapsWorstAbs, /*Fraction=*/false);
  }
}

void printExceptionHandlingStats(const BinaryContext &BC, raw_ostream &OS,
                                 iterator_range<function_iterator> &Functions) {
  std::vector<double> LPCountFractionsOfTotalBBEC;
  std::vector<double> LPCountFractionsOfTotalInvokeEC;
  for (const BinaryFunction *Function : Functions) {
    size_t LPECSum = 0;
    size_t BBECSum = 0;
    size_t InvokeECSum = 0;
    for (BinaryBasicBlock &BB : *Function) {
      const size_t BBEC = BB.getKnownExecutionCount();
      BBECSum += BBEC;
      if (BB.isLandingPad())
        LPECSum += BBEC;
      for (const MCInst &Inst : BB) {
        if (!BC.MIB->isInvoke(Inst))
          continue;
        const std::optional<MCPlus::MCLandingPad> EHInfo =
            BC.MIB->getEHInfo(Inst);
        if (EHInfo->first)
          InvokeECSum += BBEC;
      }
    }

    if (LPECSum <= MinLPECSum) {
      LPCountFractionsOfTotalBBEC.push_back(0.0);
      LPCountFractionsOfTotalInvokeEC.push_back(0.0);
      continue;
    }
    double FracTotalBBEC = 0.0;
    if (BBECSum > 0)
      FracTotalBBEC = (double)LPECSum / BBECSum;
    double FracTotalInvokeEC = 0.0;
    if (InvokeECSum > 0)
      FracTotalInvokeEC = (double)LPECSum / InvokeECSum;
    LPCountFractionsOfTotalBBEC.push_back(FracTotalBBEC);
    LPCountFractionsOfTotalInvokeEC.push_back(FracTotalInvokeEC);

    if (opts::Verbosity >= 2 && FracTotalInvokeEC >= 0.05) {
      OS << "Non-trivial usage of exception handling observed in function "
         << Function->getPrintName() << "\n"
         << formatv(
                "Fraction of total InvokeEC that goes to landing pads: {0:P}\n",
                FracTotalInvokeEC);
      if (opts::Verbosity >= 3)
        Function->dump();
    }
  }

  llvm::sort(LPCountFractionsOfTotalBBEC);
  const int RankBBEC = int(LPCountFractionsOfTotalBBEC.size() *
                           opts::PercentileForProfileQualityCheck / 100);
  llvm::sort(LPCountFractionsOfTotalInvokeEC);
  const int RankInvoke = int(LPCountFractionsOfTotalInvokeEC.size() *
                             opts::PercentileForProfileQualityCheck / 100);
  OS << formatv("exception handling usage {0:P} (of total BBEC) {1:P} (of "
                "total InvokeEC)\n",
                LPCountFractionsOfTotalBBEC[RankBBEC],
                LPCountFractionsOfTotalInvokeEC[RankInvoke]);
  if (opts::Verbosity >= 1) {
    OS << "distribution of exception handling usage as a fraction of total "
          "BBEC of each function\n";
    printDistribution(OS, LPCountFractionsOfTotalBBEC, /*Fraction=*/true);
    OS << "distribution of exception handling usage as a fraction of total "
          "InvokeEC of each function\n";
    printDistribution(OS, LPCountFractionsOfTotalInvokeEC, /*Fraction=*/true);
  }
}

void computeFlowMappings(const BinaryContext &BC, FlowInfo &TotalFlowMap) {
  // Increment block inflow and outflow with CFG jump counts.
  TotalFlowMapTy &TotalIncomingFlows = TotalFlowMap.TotalIncomingFlows;
  TotalFlowMapTy &TotalOutgoingFlows = TotalFlowMap.TotalOutgoingFlows;
  for (const auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction *Function = &BFI.second;
    std::vector<uint64_t> &IncomingFlows =
        TotalIncomingFlows[Function->getFunctionNumber()];
    std::vector<uint64_t> &OutgoingFlows =
        TotalOutgoingFlows[Function->getFunctionNumber()];
    const uint64_t NumBlocks = Function->size();
    IncomingFlows.resize(NumBlocks, 0);
    OutgoingFlows.resize(NumBlocks, 0);
    if (Function->empty() || !Function->hasValidProfile())
      continue;
    for (const BinaryBasicBlock &BB : *Function) {
      uint64_t TotalOutgoing = 0ULL;
      for (const auto &[Succ, BI] :
           llvm::zip(BB.successors(), BB.branch_info())) {
        const uint64_t Count = BI.Count;
        if (Count == BinaryBasicBlock::COUNT_NO_PROFILE || Count == 0)
          continue;
        TotalOutgoing += Count;
        IncomingFlows[Succ->getLayoutIndex()] += Count;
      }
      OutgoingFlows[BB.getLayoutIndex()] = TotalOutgoing;
    }
  }
  // Initialize TotalMaxCountMaps and TotalMinCountMaps using
  // TotalIncomingFlows and TotalOutgoingFlows
  TotalFlowMapTy &TotalMaxCountMaps = TotalFlowMap.TotalMaxCountMaps;
  TotalFlowMapTy &TotalMinCountMaps = TotalFlowMap.TotalMinCountMaps;
  for (const auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction *Function = &BFI.second;
    uint64_t FunctionNum = Function->getFunctionNumber();
    std::vector<uint64_t> &IncomingFlows = TotalIncomingFlows[FunctionNum];
    std::vector<uint64_t> &OutgoingFlows = TotalOutgoingFlows[FunctionNum];
    std::vector<uint64_t> &MaxCountMap = TotalMaxCountMaps[FunctionNum];
    std::vector<uint64_t> &MinCountMap = TotalMinCountMaps[FunctionNum];
    const uint64_t NumBlocks = Function->size();
    MaxCountMap.resize(NumBlocks, 0);
    MinCountMap.resize(NumBlocks, 0);
    if (Function->empty() || !Function->hasValidProfile())
      continue;
    for (const BinaryBasicBlock &BB : *Function) {
      uint64_t BBNum = BB.getLayoutIndex();
      MaxCountMap[BBNum] = std::max(IncomingFlows[BBNum], OutgoingFlows[BBNum]);
      MinCountMap[BBNum] = std::min(IncomingFlows[BBNum], OutgoingFlows[BBNum]);
    }
  }

  // Modify TotalMaxCountMaps and TotalMinCountMaps using call counts and
  // fill out CallGraphIncomingFlows
  FunctionFlowMapTy &CallGraphIncomingFlows =
      TotalFlowMap.CallGraphIncomingFlows;
  for (const auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction *Function = &BFI.second;
    uint64_t FunctionNum = Function->getFunctionNumber();
    std::vector<uint64_t> &MaxCountMap = TotalMaxCountMaps[FunctionNum];
    std::vector<uint64_t> &MinCountMap = TotalMinCountMaps[FunctionNum];

    // Record external entry count into CallGraphIncomingFlows
    CallGraphIncomingFlows[FunctionNum] += Function->getExternEntryCount();

    // Update MaxCountMap, MinCountMap, and CallGraphIncomingFlows
    auto recordCall = [&](const BinaryBasicBlock *SourceBB,
                          const MCSymbol *DestSymbol, uint64_t Count,
                          uint64_t TotalCallCount) {
      if (Count == BinaryBasicBlock::COUNT_NO_PROFILE)
        Count = 0;
      const BinaryFunction *DstFunc =
          DestSymbol ? BC.getFunctionForSymbol(DestSymbol) : nullptr;
      if (DstFunc)
        CallGraphIncomingFlows[DstFunc->getFunctionNumber()] += Count;
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
        for (const auto &CSI : BC.MIB->getAnnotationAs<IndirectCallSiteProfile>(
                 Inst, "CallProfile"))
          if (CSI.Symbol)
            Counts.emplace_back(CSI.Symbol, CSI.Count);
      } else {
        const uint64_t Count = BB->getExecutionCount();
        Counts.emplace_back(DstSym, Count);
      }

      return Counts;
    };

    // If the function has an invalid profile, try to use the perf data
    // directly. The call EC is only used to update CallGraphIncomingFlows.
    if (!Function->hasValidProfile() && !Function->getAllCallSites().empty()) {
      for (const IndirectCallProfile &CSI : Function->getAllCallSites())
        if (CSI.Symbol)
          recordCall(nullptr, CSI.Symbol, CSI.Count, CSI.Count);
      continue;
    } else {
      // If the function has a valid profile
      for (const BinaryBasicBlock &BB : *Function) {
        for (const MCInst &Inst : BB) {
          if (!BC.MIB->isCall(Inst))
            continue;
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
  printCFGFlowConservationStats(BC, BC.outs(), Functions, TotalFlowMap);
  printExceptionHandlingStats(BC, BC.outs(), Functions);
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
      printCFGFlowConservationStats(BC, BC.outs(), Functions, TotalFlowMap);
      printExceptionHandlingStats(BC, BC.outs(), Functions);
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
  if (ValidFunctions.empty() || opts::TopFunctionsForProfileQualityCheck == 0)
    return Error::success();

  printAll(BC, ValidFunctions, opts::TopFunctionsForProfileQualityCheck);
  return Error::success();
}

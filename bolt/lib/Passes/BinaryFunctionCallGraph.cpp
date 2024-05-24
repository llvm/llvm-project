//===- bolt/Passes/BinaryFunctionCallGraph.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BinaryFunctionCallGraph class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/BinaryFunctionCallGraph.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"
#include <stack>

#define DEBUG_TYPE "callgraph"

using namespace llvm;

namespace opts {

extern cl::opt<bool> TimeOpts;
extern cl::opt<unsigned> Verbosity;
extern cl::OptionCategory BoltCategory;

static cl::opt<std::string>
    DumpCGDot("dump-cg", cl::desc("dump callgraph to the given file"),
              cl::cat(BoltCategory));

} // namespace opts

namespace llvm {
namespace bolt {

CallGraph::NodeId BinaryFunctionCallGraph::addNode(BinaryFunction *BF,
                                                   uint32_t Size,
                                                   uint64_t Samples) {
  NodeId Id = CallGraph::addNode(Size, Samples);
  assert(size_t(Id) == Funcs.size());
  Funcs.push_back(BF);
  FuncToNodeId[BF] = Id;
  assert(Funcs[Id] == BF);
  return Id;
}

std::deque<BinaryFunction *> BinaryFunctionCallGraph::buildTraversalOrder() {
  NamedRegionTimer T1("buildcgorder", "Build cg traversal order",
                      "CG breakdown", "CG breakdown", opts::TimeOpts);
  std::deque<BinaryFunction *> TopologicalOrder;
  enum NodeStatus { NEW, VISITING, VISITED };
  std::vector<NodeStatus> NodeStatus(Funcs.size());
  std::stack<NodeId> Worklist;

  for (BinaryFunction *Func : Funcs) {
    auto It = FuncToNodeId.find(Func);
    assert(It != FuncToNodeId.end());
    const NodeId Id = It->second;
    Worklist.push(Id);
    NodeStatus[Id] = NEW;
  }

  while (!Worklist.empty()) {
    const NodeId FuncId = Worklist.top();
    Worklist.pop();

    if (NodeStatus[FuncId] == VISITED)
      continue;

    if (NodeStatus[FuncId] == VISITING) {
      TopologicalOrder.push_back(Funcs[FuncId]);
      NodeStatus[FuncId] = VISITED;
      continue;
    }

    assert(NodeStatus[FuncId] == NEW);
    NodeStatus[FuncId] = VISITING;
    Worklist.push(FuncId);
    for (const NodeId Callee : successors(FuncId)) {
      if (NodeStatus[Callee] == VISITING || NodeStatus[Callee] == VISITED)
        continue;
      Worklist.push(Callee);
    }
  }

  return TopologicalOrder;
}

BinaryFunctionCallGraph
buildCallGraph(BinaryContext &BC, CgFilterFunction Filter, bool CgFromPerfData,
               bool IncludeSplitCalls, bool UseFunctionHotSize,
               bool UseSplitHotSize, bool UseEdgeCounts,
               bool IgnoreRecursiveCalls) {
  NamedRegionTimer T1("buildcg", "Callgraph construction", "CG breakdown",
                      "CG breakdown", opts::TimeOpts);
  BinaryFunctionCallGraph Cg;
  static constexpr uint64_t COUNT_NO_PROFILE =
      BinaryBasicBlock::COUNT_NO_PROFILE;

  // Compute function size
  auto functionSize = [&](const BinaryFunction *Function) {
    return UseFunctionHotSize && Function->isSplit()
               ? Function->estimateHotSize(UseSplitHotSize)
               : Function->estimateSize();
  };

  // Add call graph nodes.
  auto lookupNode = [&](BinaryFunction *Function) {
    const CallGraph::NodeId Id = Cg.maybeGetNodeId(Function);
    if (Id == CallGraph::InvalidId) {
      // It's ok to use the hot size here when the function is split.  This is
      // because emitFunctions will emit the hot part first in the order that is
      // computed by ReorderFunctions.  The cold part will be emitted with the
      // rest of the cold functions and code.
      const size_t Size = functionSize(Function);
      // NOTE: for functions without a profile, we set the number of samples
      // to zero.  This will keep these functions from appearing in the hot
      // section.  This is a little weird because we wouldn't be trying to
      // create a node for a function unless it was the target of a call from
      // a hot block.  The alternative would be to set the count to one or
      // accumulate the number of calls from the callsite into the function
      // samples.  Results from perfomance testing seem to favor the zero
      // count though, so I'm leaving it this way for now.
      return Cg.addNode(Function, Size, Function->getKnownExecutionCount());
    }
    return Id;
  };

  // Add call graph edges.
  uint64_t NotProcessed = 0;
  uint64_t TotalCallsites = 0;
  uint64_t NoProfileCallsites = 0;
  uint64_t NumFallbacks = 0;
  uint64_t RecursiveCallsites = 0;
  for (auto &It : BC.getBinaryFunctions()) {
    BinaryFunction *Function = &It.second;

    if (Filter(*Function))
      continue;

    const CallGraph::NodeId SrcId = lookupNode(Function);
    // Offset of the current basic block from the beginning of the function
    uint64_t Offset = 0;

    auto recordCall = [&](const MCSymbol *DestSymbol, const uint64_t Count) {
      BinaryFunction *DstFunc =
          DestSymbol ? BC.getFunctionForSymbol(DestSymbol) : nullptr;
      if (!DstFunc) {
        LLVM_DEBUG(if (opts::Verbosity > 1) dbgs()
                   << "BOLT-DEBUG: buildCallGraph: no function for symbol\n");
        return false;
      }

      if (DstFunc == Function) {
        LLVM_DEBUG(dbgs() << "BOLT-INFO: recursive call detected in "
                          << *DstFunc << "\n");
        ++RecursiveCallsites;
        if (IgnoreRecursiveCalls)
          return false;
      }
      if (Filter(*DstFunc)) {
        LLVM_DEBUG(if (opts::Verbosity > 1) dbgs()
                   << "BOLT-DEBUG: buildCallGraph: filtered " << *DstFunc
                   << '\n');
        return false;
      }

      const CallGraph::NodeId DstId = lookupNode(DstFunc);
      const bool IsValidCount = Count != COUNT_NO_PROFILE;
      const uint64_t AdjCount = UseEdgeCounts && IsValidCount ? Count : 1;
      if (!IsValidCount)
        ++NoProfileCallsites;
      Cg.incArcWeight(SrcId, DstId, AdjCount, Offset);
      LLVM_DEBUG(if (opts::Verbosity > 1) {
        dbgs() << "BOLT-DEBUG: buildCallGraph: call " << *Function << " -> "
               << *DstFunc << " @ " << Offset << "\n";
      });
      return true;
    };

    // Pairs of (symbol, count) for each target at this callsite.
    using TargetDesc = std::pair<const MCSymbol *, uint64_t>;
    using CallInfoTy = std::vector<TargetDesc>;

    // Get pairs of (symbol, count) for each target at this callsite.
    // If the call is to an unknown function the symbol will be nullptr.
    // If there is no profiling data the count will be COUNT_NO_PROFILE.
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
    // directly (if requested).  If there is no perf data for this function,
    // fall back to the CFG walker which attempts to handle missing data.
    if (!Function->hasValidProfile() && CgFromPerfData &&
        !Function->getAllCallSites().empty()) {
      LLVM_DEBUG(
          dbgs() << "BOLT-DEBUG: buildCallGraph: Falling back to perf data"
                 << " for " << *Function << "\n");
      ++NumFallbacks;
      const size_t Size = functionSize(Function);
      for (const IndirectCallProfile &CSI : Function->getAllCallSites()) {
        ++TotalCallsites;

        if (!CSI.Symbol)
          continue;

        // The computed offset may exceed the hot part of the function; hence,
        // bound it by the size.
        Offset = CSI.Offset;
        if (Offset > Size)
          Offset = Size;

        if (!recordCall(CSI.Symbol, CSI.Count))
          ++NotProcessed;
      }
    } else {
      for (BinaryBasicBlock *BB : Function->getLayout().blocks()) {
        // Don't count calls from split blocks unless requested.
        if (BB->isSplit() && !IncludeSplitCalls)
          continue;

        // Determine whether the block is included in Function's (hot) size
        // See BinaryFunction::estimateHotSize
        bool BBIncludedInFunctionSize = false;
        if (UseFunctionHotSize && Function->isSplit()) {
          if (UseSplitHotSize)
            BBIncludedInFunctionSize = !BB->isSplit();
          else
            BBIncludedInFunctionSize = BB->getKnownExecutionCount() != 0;
        } else {
          BBIncludedInFunctionSize = true;
        }

        for (MCInst &Inst : *BB) {
          // Find call instructions and extract target symbols from each one.
          if (BC.MIB->isCall(Inst)) {
            const CallInfoTy CallInfo = getCallInfo(BB, Inst);

            if (!CallInfo.empty()) {
              for (const TargetDesc &CI : CallInfo) {
                ++TotalCallsites;
                if (!recordCall(CI.first, CI.second))
                  ++NotProcessed;
              }
            } else {
              ++TotalCallsites;
              ++NotProcessed;
            }
          }
          // Increase Offset if needed
          if (BBIncludedInFunctionSize)
            Offset += BC.computeCodeSize(&Inst, &Inst + 1);
        }
      }
    }
  }

#ifndef NDEBUG
  bool PrintInfo = DebugFlag && isCurrentDebugType("callgraph");
#else
  bool PrintInfo = false;
#endif
  if (PrintInfo || opts::Verbosity > 0)
    BC.outs() << format("BOLT-INFO: buildCallGraph: %u nodes, %u callsites "
                        "(%u recursive), density = %.6lf, %u callsites not "
                        "processed, %u callsites with invalid profile, "
                        "used perf data for %u stale functions.\n",
                        Cg.numNodes(), TotalCallsites, RecursiveCallsites,
                        Cg.density(), NotProcessed, NoProfileCallsites,
                        NumFallbacks);

  if (opts::DumpCGDot.getNumOccurrences()) {
    Cg.printDot(opts::DumpCGDot, [&](CallGraph::NodeId Id) {
      return Cg.nodeIdToFunc(Id)->getPrintName();
    });
  }

  return Cg;
}

} // namespace bolt
} // namespace llvm

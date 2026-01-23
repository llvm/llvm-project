//===- bolt/Passes/FrameOptimizer.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FrameOptimizerPass class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/FrameOptimizer.h"
#include "bolt/Core/BinaryFunctionCallGraph.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Passes/DataflowInfoManager.h"
#include "bolt/Passes/ShrinkWrapping.h"
#include "bolt/Passes/StackAvailableExpressions.h"
#include "bolt/Passes/StackReachingUses.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/Support/Timer.h"
#include <deque>

#define DEBUG_TYPE "fop"

using namespace llvm;

namespace opts {
extern cl::opt<unsigned> Verbosity;
extern cl::opt<bool> TimeOpts;
extern cl::OptionCategory BoltOptCategory;

using namespace bolt;

cl::opt<FrameOptimizationType>
FrameOptimization("frame-opt",
  cl::init(FOP_NONE),
  cl::desc("optimize stack frame accesses"),
  cl::values(
    clEnumValN(FOP_NONE, "none", "do not perform frame optimization"),
    clEnumValN(FOP_HOT, "hot", "perform FOP on hot functions"),
    clEnumValN(FOP_ALL, "all", "perform FOP on all functions")),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool> RemoveStores(
    "frame-opt-rm-stores", cl::init(FOP_NONE),
    cl::desc("apply additional analysis to remove stores (experimental)"),
    cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {

void FrameOptimizerPass::removeUnnecessaryLoads(const RegAnalysis &RA,
                                                const FrameAnalysis &FA,
                                                BinaryFunction &BF) {
  StackAvailableExpressions SAE(RA, FA, BF);
  SAE.run();

  LLVM_DEBUG(dbgs() << "Performing unnecessary loads removal\n");
  std::deque<std::pair<BinaryBasicBlock *, MCInst *>> ToErase;
  bool Changed = false;
  const auto ExprEnd = SAE.expr_end();
  MCPlusBuilder *MIB = BF.getBinaryContext().MIB.get();
  for (BinaryBasicBlock &BB : BF) {
    LLVM_DEBUG(dbgs() << "\tNow at BB " << BB.getName() << "\n");
    const MCInst *Prev = nullptr;
    for (MCInst &Inst : BB) {
      LLVM_DEBUG({
        dbgs() << "\t\tNow at ";
        Inst.dump();
        for (auto I = Prev ? SAE.expr_begin(*Prev) : SAE.expr_begin(BB);
             I != ExprEnd; ++I) {
          dbgs() << "\t\t\tReached by: ";
          (*I)->dump();
        }
      });
      // if Inst is a load from stack and the current available expressions show
      // this value is available in a register or immediate, replace this load
      // with move from register or from immediate.
      ErrorOr<const FrameIndexEntry &> FIEX = FA.getFIEFor(Inst);
      if (!FIEX) {
        Prev = &Inst;
        continue;
      }
      // FIXME: Change to remove IsSimple == 0. We're being conservative here,
      // but once replaceMemOperandWithReg is ready, we should feed it with all
      // sorts of complex instructions.
      if (FIEX->IsLoad == false || FIEX->IsSimple == false ||
          FIEX->StackOffset >= 0) {
        Prev = &Inst;
        continue;
      }

      for (auto I = Prev ? SAE.expr_begin(*Prev) : SAE.expr_begin(BB);
           I != ExprEnd; ++I) {
        const MCInst *AvailableInst = *I;
        ErrorOr<const FrameIndexEntry &> FIEY = FA.getFIEFor(*AvailableInst);
        if (!FIEY)
          continue;
        assert(FIEY->IsStore && FIEY->IsSimple);
        if (FIEX->StackOffset != FIEY->StackOffset || FIEX->Size != FIEY->Size)
          continue;
        // TODO: Change push/pops to stack adjustment instruction
        if (MIB->isPop(Inst))
          continue;

        ++NumRedundantLoads;
        FreqRedundantLoads += BB.getKnownExecutionCount();
        Changed = true;
        LLVM_DEBUG(dbgs() << "Redundant load instruction: ");
        LLVM_DEBUG(Inst.dump());
        LLVM_DEBUG(dbgs() << "Related store instruction: ");
        LLVM_DEBUG(AvailableInst->dump());
        LLVM_DEBUG(dbgs() << "@BB: " << BB.getName() << "\n");
        // Replace load
        if (FIEY->IsStoreFromReg) {
          if (!MIB->replaceMemOperandWithReg(Inst, FIEY->RegOrImm)) {
            LLVM_DEBUG(dbgs() << "FAILED to change operand to a reg\n");
            break;
          }
          FreqLoadsChangedToReg += BB.getKnownExecutionCount();
          MIB->removeAnnotation(Inst, "FrameAccessEntry");
          LLVM_DEBUG(dbgs() << "Changed operand to a reg\n");
          if (MIB->isRedundantMove(Inst)) {
            ++NumLoadsDeleted;
            FreqLoadsDeleted += BB.getKnownExecutionCount();
            LLVM_DEBUG(dbgs() << "Created a redundant move\n");
            // Delete it!
            ToErase.push_front(std::make_pair(&BB, &Inst));
          }
        } else {
          char Buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
          support::ulittle64_t::ref(Buf + 0) = FIEY->RegOrImm;
          LLVM_DEBUG(dbgs() << "Changing operand to an imm... ");
          if (!MIB->replaceMemOperandWithImm(Inst, StringRef(Buf, 8), 0)) {
            LLVM_DEBUG(dbgs() << "FAILED\n");
          } else {
            FreqLoadsChangedToImm += BB.getKnownExecutionCount();
            MIB->removeAnnotation(Inst, "FrameAccessEntry");
            LLVM_DEBUG(dbgs() << "Ok\n");
          }
        }
        LLVM_DEBUG(dbgs() << "Changed to: ");
        LLVM_DEBUG(Inst.dump());
        break;
      }
      Prev = &Inst;
    }
  }
  if (Changed)
    LLVM_DEBUG(dbgs() << "FOP modified \"" << BF.getPrintName() << "\"\n");

  // TODO: Implement an interface of eraseInstruction that works out the
  // complete list of elements to remove.
  for (std::pair<BinaryBasicBlock *, MCInst *> I : ToErase)
    I.first->eraseInstruction(I.first->findInstruction(I.second));
}

void FrameOptimizerPass::removeUnusedStores(const FrameAnalysis &FA,
                                            BinaryFunction &BF) {
  StackReachingUses SRU(FA, BF);
  SRU.run();

  LLVM_DEBUG(dbgs() << "Performing unused stores removal\n");
  std::vector<std::pair<BinaryBasicBlock *, MCInst *>> ToErase;
  bool Changed = false;
  for (BinaryBasicBlock &BB : BF) {
    LLVM_DEBUG(dbgs() << "\tNow at BB " << BB.getName() << "\n");
    const MCInst *Prev = nullptr;
    for (MCInst &Inst : llvm::reverse(BB)) {
      LLVM_DEBUG({
        dbgs() << "\t\tNow at ";
        Inst.dump();
        for (auto I = Prev ? SRU.expr_begin(*Prev) : SRU.expr_begin(BB);
             I != SRU.expr_end(); ++I) {
          dbgs() << "\t\t\tReached by: ";
          (*I)->dump();
        }
      });
      ErrorOr<const FrameIndexEntry &> FIEX = FA.getFIEFor(Inst);
      if (!FIEX) {
        Prev = &Inst;
        continue;
      }
      if (FIEX->IsLoad || !FIEX->IsSimple || FIEX->StackOffset >= 0) {
        Prev = &Inst;
        continue;
      }

      if (SRU.isStoreUsed(*FIEX,
                          Prev ? SRU.expr_begin(*Prev) : SRU.expr_begin(BB))) {
        Prev = &Inst;
        continue;
      }
      // TODO: Change push/pops to stack adjustment instruction
      if (BF.getBinaryContext().MIB->isPush(Inst))
        continue;

      ++NumRedundantStores;
      FreqRedundantStores += BB.getKnownExecutionCount();
      Changed = true;
      LLVM_DEBUG(dbgs() << "Unused store instruction: ");
      LLVM_DEBUG(Inst.dump());
      LLVM_DEBUG(dbgs() << "@BB: " << BB.getName() << "\n");
      LLVM_DEBUG(dbgs() << "FIE offset = " << FIEX->StackOffset
                        << " size = " << (int)FIEX->Size << "\n");
      // Delete it!
      ToErase.emplace_back(&BB, &Inst);
      Prev = &Inst;
    }
  }

  for (std::pair<BinaryBasicBlock *, MCInst *> I : ToErase)
    I.first->eraseInstruction(I.first->findInstruction(I.second));

  if (Changed)
    LLVM_DEBUG(dbgs() << "FOP modified \"" << BF.getPrintName() << "\"\n");
}

Error FrameOptimizerPass::runOnFunctions(BinaryContext &BC) {
  if (opts::FrameOptimization == FOP_NONE)
    return Error::success();

  if (!BC.isX86()) {
    BC.errs() << "BOLT-ERROR: " << getName() << " is supported only on X86\n";
    exit(1);
  }

  std::unique_ptr<BinaryFunctionCallGraph> CG;
  std::unique_ptr<FrameAnalysis> FA;
  std::unique_ptr<RegAnalysis> RA;

  {
    NamedRegionTimer T1("callgraph", "create call graph", "FOP",
                        "FOP breakdown", opts::TimeOpts);
    CG = std::make_unique<BinaryFunctionCallGraph>(buildCallGraph(BC));
  }

  {
    NamedRegionTimer T1("frameanalysis", "frame analysis", "FOP",
                        "FOP breakdown", opts::TimeOpts);
    FA = std::make_unique<FrameAnalysis>(BC, *CG);
  }

  {
    NamedRegionTimer T1("reganalysis", "reg analysis", "FOP", "FOP breakdown",
                        opts::TimeOpts);
    RA = std::make_unique<RegAnalysis>(BC, &BC.getBinaryFunctions(), CG.get());
  }

  // Perform caller-saved register optimizations, then callee-saved register
  // optimizations (shrink wrapping)
  for (auto &I : BC.getBinaryFunctions()) {
    if (!FA->hasFrameInfo(I.second))
      continue;
    // Restrict pass execution if user asked to only run on hot functions
    if (opts::FrameOptimization == FOP_HOT) {
      if (I.second.getKnownExecutionCount() < BC.getHotThreshold())
        continue;
      LLVM_DEBUG(
          dbgs() << "Considering " << I.second.getPrintName()
                 << " for frame optimizations because its execution count ( "
                 << I.second.getKnownExecutionCount()
                 << " ) exceeds our hotness threshold ( "
                 << BC.getHotThreshold() << " )\n");
    }

    {
      NamedRegionTimer T1("removeloads", "remove loads", "FOP", "FOP breakdown",
                          opts::TimeOpts);
      if (!FA->hasStackArithmetic(I.second))
        removeUnnecessaryLoads(*RA, *FA, I.second);
    }

    if (opts::RemoveStores) {
      NamedRegionTimer T1("removestores", "remove stores", "FOP",
                          "FOP breakdown", opts::TimeOpts);
      if (!FA->hasStackArithmetic(I.second))
        removeUnusedStores(*FA, I.second);
    }
    // Don't even start shrink wrapping if no profiling info is available
    if (I.second.getKnownExecutionCount() == 0)
      continue;
  }

  {
    NamedRegionTimer T1("shrinkwrapping", "shrink wrapping", "FOP",
                        "FOP breakdown", opts::TimeOpts);
    if (Error E = performShrinkWrapping(*RA, *FA, BC))
      return Error(std::move(E));
  }

  BC.outs() << "BOLT-INFO: FOP optimized " << NumRedundantLoads
            << " redundant load(s) and " << NumRedundantStores
            << " unused store(s)\n";
  BC.outs() << "BOLT-INFO: Frequency of redundant loads is "
            << FreqRedundantLoads << " and frequency of unused stores is "
            << FreqRedundantStores << "\n";
  BC.outs() << "BOLT-INFO: Frequency of loads changed to use a register is "
            << FreqLoadsChangedToReg
            << " and frequency of loads changed to use an immediate is "
            << FreqLoadsChangedToImm << "\n";
  BC.outs() << "BOLT-INFO: FOP deleted " << NumLoadsDeleted
            << " load(s) (dyn count: " << FreqLoadsDeleted << ") and "
            << NumRedundantStores << " store(s)\n";
  FA->printStats();
  ShrinkWrapping::printStats(BC);
  return Error::success();
}

Error FrameOptimizerPass::performShrinkWrapping(const RegAnalysis &RA,
                                                const FrameAnalysis &FA,
                                                BinaryContext &BC) {
  // Initialize necessary annotations to allow safe parallel accesses to
  // annotation index in MIB
  BC.MIB->getOrCreateAnnotationIndex(CalleeSavedAnalysis::getSaveTagName());
  BC.MIB->getOrCreateAnnotationIndex(CalleeSavedAnalysis::getRestoreTagName());
  BC.MIB->getOrCreateAnnotationIndex(StackLayoutModifier::getTodoTagName());
  BC.MIB->getOrCreateAnnotationIndex(StackLayoutModifier::getSlotTagName());
  BC.MIB->getOrCreateAnnotationIndex(
      StackLayoutModifier::getOffsetCFIRegTagName());
  BC.MIB->getOrCreateAnnotationIndex("ReachingDefs");
  BC.MIB->getOrCreateAnnotationIndex("ReachingUses");
  BC.MIB->getOrCreateAnnotationIndex("LivenessAnalysis");
  BC.MIB->getOrCreateAnnotationIndex("StackReachingUses");
  BC.MIB->getOrCreateAnnotationIndex("PostDominatorAnalysis");
  BC.MIB->getOrCreateAnnotationIndex("DominatorAnalysis");
  BC.MIB->getOrCreateAnnotationIndex("StackPointerTracking");
  BC.MIB->getOrCreateAnnotationIndex("StackPointerTrackingForInternalCalls");
  BC.MIB->getOrCreateAnnotationIndex("StackAvailableExpressions");
  BC.MIB->getOrCreateAnnotationIndex("StackAllocationAnalysis");
  BC.MIB->getOrCreateAnnotationIndex("ShrinkWrap-Todo");
  BC.MIB->getOrCreateAnnotationIndex("PredictiveStackPointerTracking");
  BC.MIB->getOrCreateAnnotationIndex("ReachingInsnsBackward");
  BC.MIB->getOrCreateAnnotationIndex("ReachingInsns");
  BC.MIB->getOrCreateAnnotationIndex("AccessesDeletedPos");
  BC.MIB->getOrCreateAnnotationIndex("DeleteMe");

  std::vector<std::pair<uint64_t, const BinaryFunction *>> Top10Funcs;
  auto LogFunc = [&](BinaryFunction &BF) {
    auto Lower = llvm::lower_bound(
        Top10Funcs, BF.getKnownExecutionCount(),
        [](const std::pair<uint64_t, const BinaryFunction *> &Elmt,
           uint64_t Value) { return Elmt.first > Value; });
    if (Lower == Top10Funcs.end() && Top10Funcs.size() >= 10)
      return;
    Top10Funcs.insert(Lower,
                      std::make_pair<>(BF.getKnownExecutionCount(), &BF));
    if (Top10Funcs.size() > 10)
      Top10Funcs.resize(10);
  };
  (void)LogFunc;

  ParallelUtilities::PredicateTy SkipPredicate = [&](const BinaryFunction &BF) {
    if (BF.getFunctionScore() == 0)
      return true;

    return false;
  };

  const bool HotOnly = opts::FrameOptimization == FOP_HOT;

  Error SWError = Error::success();

  ParallelUtilities::WorkFuncWithAllocTy WorkFunction =
      [&](BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocatorId) {
        DataflowInfoManager Info(BF, &RA, &FA, AllocatorId);
        ShrinkWrapping SW(FA, BF, Info, AllocatorId);

        auto ChangedOrErr = SW.perform(HotOnly);
        if (auto E = ChangedOrErr.takeError()) {
          std::lock_guard<std::mutex> Lock(FuncsChangedMutex);
          SWError = joinErrors(std::move(SWError), Error(std::move(E)));
          return;
        }
        const bool Changed = *ChangedOrErr;
        if (Changed) {
          std::lock_guard<std::mutex> Lock(FuncsChangedMutex);
          FuncsChanged.insert(&BF);
          LLVM_DEBUG(LogFunc(BF));
        }
      };

  ParallelUtilities::runOnEachFunctionWithUniqueAllocId(
      BC, ParallelUtilities::SchedulingPolicy::SP_INST_QUADRATIC, WorkFunction,
      SkipPredicate, "shrink-wrapping");

  if (!Top10Funcs.empty()) {
    BC.outs() << "BOLT-INFO: top 10 functions changed by shrink wrapping:\n";
    for (const auto &Elmt : Top10Funcs)
      BC.outs() << Elmt.first << " : " << Elmt.second->getPrintName() << "\n";
  }
  return SWError;
}

} // namespace bolt
} // namespace llvm

//===- bolt/Passes/Inliner.cpp - Inlining pass for low-level binary IR ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Inliner class used for inlining binary functions.
//
// The current inliner has a limited callee support
// (see Inliner::getInliningInfo() for the most up-to-date details):
//
//  * No exception handling
//  * No jump tables
//  * Single entry point
//  * CFI update not supported - breaks unwinding
//  * Regular Call Sites:
//    - only leaf functions (or callees with only tail calls)
//      * no invokes (they can't be tail calls)
//    - no direct use of %rsp
//  * Tail Call Sites:
//    - since the stack is unmodified, the regular call limitations are lifted
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/Inliner.h"
#include "bolt/Core/MCPlus.h"
#include "llvm/Support/CommandLine.h"
#include <map>

#define DEBUG_TYPE "bolt-inliner"

using namespace llvm;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

static cl::opt<bool>
    AdjustProfile("inline-ap",
                  cl::desc("adjust function profile after inlining"),
                  cl::cat(BoltOptCategory));

static cl::list<std::string>
ForceInlineFunctions("force-inline",
  cl::CommaSeparated,
  cl::desc("list of functions to always consider for inlining"),
  cl::value_desc("func1,func2,func3,..."),
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool> InlineAll("inline-all", cl::desc("inline all functions"),
                               cl::cat(BoltOptCategory));

static cl::opt<bool> InlineIgnoreLeafCFI(
    "inline-ignore-leaf-cfi",
    cl::desc("inline leaf functions with CFI programs (can break unwinding)"),
    cl::init(true), cl::ReallyHidden, cl::cat(BoltOptCategory));

static cl::opt<bool> InlineIgnoreCFI(
    "inline-ignore-cfi",
    cl::desc(
        "inline functions with CFI programs (can break exception handling)"),
    cl::ReallyHidden, cl::cat(BoltOptCategory));

static cl::opt<unsigned>
    InlineLimit("inline-limit",
                cl::desc("maximum number of call sites to inline"), cl::init(0),
                cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<unsigned>
    InlineMaxIters("inline-max-iters",
                   cl::desc("maximum number of inline iterations"), cl::init(3),
                   cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool> InlineSmallFunctions(
    "inline-small-functions",
    cl::desc("inline functions if increase in size is less than defined by "
             "-inline-small-functions-bytes"),
    cl::cat(BoltOptCategory));

static cl::opt<unsigned> InlineSmallFunctionsBytes(
    "inline-small-functions-bytes",
    cl::desc("max number of bytes for the function to be considered small for "
             "inlining purposes"),
    cl::init(4), cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool> NoInline(
    "no-inline",
    cl::desc("disable all inlining (overrides other inlining options)"),
    cl::cat(BoltOptCategory));

/// This function returns true if any of inlining options are specified and the
/// inlining pass should be executed. Whenever a new inlining option is added,
/// this function should reflect the change.
bool inliningEnabled() {
  return !NoInline &&
         (InlineAll || InlineSmallFunctions || !ForceInlineFunctions.empty());
}

bool mustConsider(const llvm::bolt::BinaryFunction &Function) {
  for (std::string &Name : opts::ForceInlineFunctions)
    if (Function.hasName(Name))
      return true;
  return false;
}

void syncOptions() {
  if (opts::InlineIgnoreCFI)
    opts::InlineIgnoreLeafCFI = true;

  if (opts::InlineAll)
    opts::InlineSmallFunctions = true;
}

} // namespace opts

namespace llvm {
namespace bolt {

uint64_t Inliner::SizeOfCallInst;
uint64_t Inliner::SizeOfTailCallInst;

uint64_t Inliner::getSizeOfCallInst(const BinaryContext &BC) {
  if (SizeOfCallInst)
    return SizeOfCallInst;

  MCInst Inst;
  BC.MIB->createCall(Inst, BC.Ctx->createNamedTempSymbol(), BC.Ctx.get());
  SizeOfCallInst = BC.computeInstructionSize(Inst);

  return SizeOfCallInst;
}

uint64_t Inliner::getSizeOfTailCallInst(const BinaryContext &BC) {
  if (SizeOfTailCallInst)
    return SizeOfTailCallInst;

  MCInst Inst;
  BC.MIB->createTailCall(Inst, BC.Ctx->createNamedTempSymbol(), BC.Ctx.get());
  SizeOfTailCallInst = BC.computeInstructionSize(Inst);

  return SizeOfTailCallInst;
}

InliningInfo getInliningInfo(const BinaryFunction &BF) {
  const BinaryContext &BC = BF.getBinaryContext();
  bool DirectSP = false;
  bool HasCFI = false;
  bool IsLeaf = true;

  // Perform necessary checks unless the option overrides it.
  if (!opts::mustConsider(BF)) {
    if (BF.hasSDTMarker())
      return INL_NONE;

    if (BF.hasEHRanges())
      return INL_NONE;

    if (BF.isMultiEntry())
      return INL_NONE;

    if (BF.hasJumpTables())
      return INL_NONE;

    const MCPhysReg SPReg = BC.MIB->getStackPointer();
    for (const BinaryBasicBlock *BB : BF.layout()) {
      for (const MCInst &Inst : *BB) {
        // Tail calls are marked as implicitly using the stack pointer and they
        // could be inlined.
        if (BC.MIB->isTailCall(Inst))
          break;

        if (BC.MIB->isCFI(Inst)) {
          HasCFI = true;
          continue;
        }

        if (BC.MIB->isCall(Inst))
          IsLeaf = false;

        // Push/pop instructions are straightforward to handle.
        if (BC.MIB->isPush(Inst) || BC.MIB->isPop(Inst))
          continue;

        DirectSP |= BC.MIB->hasDefOfPhysReg(Inst, SPReg) ||
                    BC.MIB->hasUseOfPhysReg(Inst, SPReg);
      }
    }
  }

  if (HasCFI) {
    if (!opts::InlineIgnoreLeafCFI)
      return INL_NONE;

    if (!IsLeaf && !opts::InlineIgnoreCFI)
      return INL_NONE;
  }

  InliningInfo Info(DirectSP ? INL_TAILCALL : INL_ANY);

  size_t Size = BF.estimateSize();

  Info.SizeAfterInlining = Size;
  Info.SizeAfterTailCallInlining = Size;

  // Handle special case of the known size reduction.
  if (BF.size() == 1) {
    // For a regular call the last return instruction could be removed
    // (or converted to a branch).
    const MCInst *LastInst = BF.back().getLastNonPseudoInstr();
    if (LastInst && BC.MIB->isReturn(*LastInst) &&
        !BC.MIB->isTailCall(*LastInst)) {
      const uint64_t RetInstSize = BC.computeInstructionSize(*LastInst);
      assert(Size >= RetInstSize);
      Info.SizeAfterInlining -= RetInstSize;
    }
  }

  return Info;
}

void Inliner::findInliningCandidates(BinaryContext &BC) {
  for (const auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction &Function = BFI.second;
    if (!shouldOptimize(Function))
      continue;
    const InliningInfo InlInfo = getInliningInfo(Function);
    if (InlInfo.Type != INL_NONE)
      InliningCandidates[&Function] = InlInfo;
  }
}

std::pair<BinaryBasicBlock *, BinaryBasicBlock::iterator>
Inliner::inlineCall(BinaryBasicBlock &CallerBB,
                    BinaryBasicBlock::iterator CallInst,
                    const BinaryFunction &Callee) {
  BinaryFunction &CallerFunction = *CallerBB.getFunction();
  BinaryContext &BC = CallerFunction.getBinaryContext();
  auto &MIB = *BC.MIB;

  assert(MIB.isCall(*CallInst) && "can only inline a call or a tail call");
  assert(!Callee.isMultiEntry() &&
         "cannot inline function with multiple entries");
  assert(!Callee.hasJumpTables() &&
         "cannot inline function with jump table(s)");

  // Get information about the call site.
  const bool CSIsInvoke = BC.MIB->isInvoke(*CallInst);
  const bool CSIsTailCall = BC.MIB->isTailCall(*CallInst);
  const int64_t CSGNUArgsSize = BC.MIB->getGnuArgsSize(*CallInst);
  const Optional<MCPlus::MCLandingPad> CSEHInfo = BC.MIB->getEHInfo(*CallInst);

  // Split basic block at the call site if there will be more incoming edges
  // coming from the callee.
  BinaryBasicBlock *FirstInlinedBB = &CallerBB;
  if (Callee.front().pred_size() && CallInst != CallerBB.begin()) {
    FirstInlinedBB = CallerBB.splitAt(CallInst);
    CallInst = FirstInlinedBB->begin();
  }

  // Split basic block after the call instruction unless the callee is trivial
  // (i.e. consists of a single basic block). If necessary, obtain a basic block
  // for return instructions in the callee to redirect to.
  BinaryBasicBlock *NextBB = nullptr;
  if (Callee.size() > 1) {
    if (std::next(CallInst) != FirstInlinedBB->end())
      NextBB = FirstInlinedBB->splitAt(std::next(CallInst));
    else
      NextBB = FirstInlinedBB->getSuccessor();
  }
  if (NextBB)
    FirstInlinedBB->removeSuccessor(NextBB);

  // Remove the call instruction.
  auto InsertII = FirstInlinedBB->eraseInstruction(CallInst);

  double ProfileRatio = 0;
  if (uint64_t CalleeExecCount = Callee.getKnownExecutionCount())
    ProfileRatio =
        (double)FirstInlinedBB->getKnownExecutionCount() / CalleeExecCount;

  // Save execution count of the first block as we don't want it to change
  // later due to profile adjustment rounding errors.
  const uint64_t FirstInlinedBBCount = FirstInlinedBB->getKnownExecutionCount();

  // Copy basic blocks and maintain a map from their origin.
  std::unordered_map<const BinaryBasicBlock *, BinaryBasicBlock *> InlinedBBMap;
  InlinedBBMap[&Callee.front()] = FirstInlinedBB;
  for (auto BBI = std::next(Callee.begin()); BBI != Callee.end(); ++BBI) {
    BinaryBasicBlock *InlinedBB = CallerFunction.addBasicBlock();
    InlinedBBMap[&*BBI] = InlinedBB;
    InlinedBB->setCFIState(FirstInlinedBB->getCFIState());
    if (Callee.hasValidProfile())
      InlinedBB->setExecutionCount(BBI->getKnownExecutionCount());
    else
      InlinedBB->setExecutionCount(FirstInlinedBBCount);
  }

  // Copy over instructions and edges.
  for (const BinaryBasicBlock &BB : Callee) {
    BinaryBasicBlock *InlinedBB = InlinedBBMap[&BB];

    if (InlinedBB != FirstInlinedBB)
      InsertII = InlinedBB->begin();

    // Copy over instructions making any necessary mods.
    for (MCInst Inst : BB) {
      if (MIB.isPseudo(Inst))
        continue;

      MIB.stripAnnotations(Inst, /*KeepTC=*/BC.isX86());

      // Fix branch target. Strictly speaking, we don't have to do this as
      // targets of direct branches will be fixed later and don't matter
      // in the CFG state. However, disassembly may look misleading, and
      // hence we do the fixing.
      if (MIB.isBranch(Inst)) {
        assert(!MIB.isIndirectBranch(Inst) &&
               "unexpected indirect branch in callee");
        const BinaryBasicBlock *TargetBB =
            Callee.getBasicBlockForLabel(MIB.getTargetSymbol(Inst));
        assert(TargetBB && "cannot find target block in callee");
        MIB.replaceBranchTarget(Inst, InlinedBBMap[TargetBB]->getLabel(),
                                BC.Ctx.get());
      }

      if (CSIsTailCall || (!MIB.isCall(Inst) && !MIB.isReturn(Inst))) {
        InsertII =
            std::next(InlinedBB->insertInstruction(InsertII, std::move(Inst)));
        continue;
      }

      // Handle special instructions for a non-tail call site.
      if (!MIB.isCall(Inst)) {
        // Returns are removed.
        break;
      }

      MIB.convertTailCallToCall(Inst);

      // Propagate EH-related info to call instructions.
      if (CSIsInvoke) {
        MIB.addEHInfo(Inst, *CSEHInfo);
        if (CSGNUArgsSize >= 0)
          MIB.addGnuArgsSize(Inst, CSGNUArgsSize);
      }

      InsertII =
          std::next(InlinedBB->insertInstruction(InsertII, std::move(Inst)));
    }

    // Add CFG edges to the basic blocks of the inlined instance.
    std::vector<BinaryBasicBlock *> Successors(BB.succ_size());
    llvm::transform(BB.successors(), Successors.begin(),
                    [&InlinedBBMap](const BinaryBasicBlock *BB) {
                      return InlinedBBMap.at(BB);
                    });

    if (CallerFunction.hasValidProfile() && Callee.hasValidProfile())
      InlinedBB->addSuccessors(Successors.begin(), Successors.end(),
                               BB.branch_info_begin(), BB.branch_info_end());
    else
      InlinedBB->addSuccessors(Successors.begin(), Successors.end());

    if (!CSIsTailCall && BB.succ_size() == 0 && NextBB) {
      // Either it's a return block or the last instruction never returns.
      InlinedBB->addSuccessor(NextBB, InlinedBB->getExecutionCount());
    }

    // Scale profiling info for blocks and edges after inlining.
    if (CallerFunction.hasValidProfile() && Callee.size() > 1) {
      if (opts::AdjustProfile)
        InlinedBB->adjustExecutionCount(ProfileRatio);
      else
        InlinedBB->setExecutionCount(InlinedBB->getKnownExecutionCount() *
                                     ProfileRatio);
    }
  }

  // Restore the original execution count of the first inlined basic block.
  FirstInlinedBB->setExecutionCount(FirstInlinedBBCount);

  CallerFunction.recomputeLandingPads();

  if (NextBB)
    return std::make_pair(NextBB, NextBB->begin());

  if (Callee.size() == 1)
    return std::make_pair(FirstInlinedBB, InsertII);

  return std::make_pair(FirstInlinedBB, FirstInlinedBB->end());
}

bool Inliner::inlineCallsInFunction(BinaryFunction &Function) {
  BinaryContext &BC = Function.getBinaryContext();
  std::vector<BinaryBasicBlock *> Blocks(Function.layout().begin(),
                                         Function.layout().end());
  llvm::sort(
      Blocks, [](const BinaryBasicBlock *BB1, const BinaryBasicBlock *BB2) {
        return BB1->getKnownExecutionCount() > BB2->getKnownExecutionCount();
      });

  bool DidInlining = false;
  for (BinaryBasicBlock *BB : Blocks) {
    for (auto InstIt = BB->begin(); InstIt != BB->end();) {
      MCInst &Inst = *InstIt;
      if (!BC.MIB->isCall(Inst) || MCPlus::getNumPrimeOperands(Inst) != 1 ||
          !Inst.getOperand(0).isExpr()) {
        ++InstIt;
        continue;
      }

      const MCSymbol *TargetSymbol = BC.MIB->getTargetSymbol(Inst);
      assert(TargetSymbol && "target symbol expected for direct call");

      // Don't inline calls to a secondary entry point in a target function.
      uint64_t EntryID = 0;
      BinaryFunction *TargetFunction =
          BC.getFunctionForSymbol(TargetSymbol, &EntryID);
      if (!TargetFunction || EntryID != 0) {
        ++InstIt;
        continue;
      }

      // Don't do recursive inlining.
      if (TargetFunction == &Function) {
        ++InstIt;
        continue;
      }

      auto IInfo = InliningCandidates.find(TargetFunction);
      if (IInfo == InliningCandidates.end()) {
        ++InstIt;
        continue;
      }

      const bool IsTailCall = BC.MIB->isTailCall(Inst);
      if (!IsTailCall && IInfo->second.Type == INL_TAILCALL) {
        ++InstIt;
        continue;
      }

      int64_t SizeAfterInlining;
      if (IsTailCall)
        SizeAfterInlining =
            IInfo->second.SizeAfterTailCallInlining - getSizeOfTailCallInst(BC);
      else
        SizeAfterInlining =
            IInfo->second.SizeAfterInlining - getSizeOfCallInst(BC);

      if (!opts::InlineAll && !opts::mustConsider(*TargetFunction)) {
        if (!opts::InlineSmallFunctions ||
            SizeAfterInlining > opts::InlineSmallFunctionsBytes) {
          ++InstIt;
          continue;
        }
      }

      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: inlining call to " << *TargetFunction
                        << " in " << Function << " : " << BB->getName()
                        << ". Count: " << BB->getKnownExecutionCount()
                        << ". Size change: " << SizeAfterInlining
                        << " bytes.\n");

      std::tie(BB, InstIt) = inlineCall(*BB, InstIt, *TargetFunction);

      DidInlining = true;
      TotalInlinedBytes += SizeAfterInlining;

      ++NumInlinedCallSites;
      NumInlinedDynamicCalls += BB->getExecutionCount();

      // Subtract basic block execution count from the callee execution count.
      if (opts::AdjustProfile)
        TargetFunction->adjustExecutionCount(BB->getKnownExecutionCount());

      // Check if the caller inlining status has to be adjusted.
      if (IInfo->second.Type == INL_TAILCALL) {
        auto CallerIInfo = InliningCandidates.find(&Function);
        if (CallerIInfo != InliningCandidates.end() &&
            CallerIInfo->second.Type == INL_ANY) {
          LLVM_DEBUG(dbgs() << "adjusting inlining status for function "
                            << Function << '\n');
          CallerIInfo->second.Type = INL_TAILCALL;
        }
      }

      if (NumInlinedCallSites == opts::InlineLimit)
        return true;
    }
  }

  return DidInlining;
}

void Inliner::runOnFunctions(BinaryContext &BC) {
  opts::syncOptions();

  if (!opts::inliningEnabled())
    return;

  bool InlinedOnce;
  unsigned NumIters = 0;
  do {
    if (opts::InlineLimit && NumInlinedCallSites >= opts::InlineLimit)
      break;

    InlinedOnce = false;

    InliningCandidates.clear();
    findInliningCandidates(BC);

    std::vector<BinaryFunction *> ConsideredFunctions;
    for (auto &BFI : BC.getBinaryFunctions()) {
      BinaryFunction &Function = BFI.second;
      if (!shouldOptimize(Function))
        continue;
      ConsideredFunctions.push_back(&Function);
    }
    llvm::sort(ConsideredFunctions, [](const BinaryFunction *A,
                                       const BinaryFunction *B) {
      return B->getKnownExecutionCount() < A->getKnownExecutionCount();
    });
    for (BinaryFunction *Function : ConsideredFunctions) {
      if (opts::InlineLimit && NumInlinedCallSites >= opts::InlineLimit)
        break;

      const bool DidInline = inlineCallsInFunction(*Function);

      if (DidInline)
        Modified.insert(Function);

      InlinedOnce |= DidInline;
    }

    ++NumIters;
  } while (InlinedOnce && NumIters < opts::InlineMaxIters);

  if (NumInlinedCallSites)
    outs() << "BOLT-INFO: inlined " << NumInlinedDynamicCalls << " calls at "
           << NumInlinedCallSites << " call sites in " << NumIters
           << " iteration(s). Change in binary size: " << TotalInlinedBytes
           << " bytes.\n";
}

} // namespace bolt
} // namespace llvm

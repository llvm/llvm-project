//===- bolt/Passes/PointerAuthCFIFixup.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the PointerAuthCFIFixup class. It inserts
// OpNegateRAState CFIs to places where the state of two consecutive
// instructions are different.
//
//===----------------------------------------------------------------------===//
#include "bolt/Passes/PointerAuthCFIFixup.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/ParallelUtilities.h"
#include <cstdlib>

using namespace llvm;

namespace llvm {
namespace bolt {

static bool PassFailed = false;

void PointerAuthCFIFixup::runOnFunction(BinaryFunction &BF) {
  if (PassFailed)
    return;

  BinaryContext &BC = BF.getBinaryContext();

  if (BF.getState() == BinaryFunction::State::Empty)
    return;

  if (BF.getState() != BinaryFunction::State::CFG &&
      BF.getState() != BinaryFunction::State::CFG_Finalized) {
    BC.outs() << "BOLT-INFO: no CFG for " << BF.getPrintName()
              << " in PointerAuthCFIFixup\n";
    return;
  }

  inferUnknownStates(BF);

  for (FunctionFragment &FF : BF.getLayout().fragments()) {
    coverFunctionFragmentStart(BF, FF);
    bool FirstIter = true;
    bool PrevRAState = false;
    // As this pass runs after function splitting, we should only check
    // consecutive instructions inside FunctionFragments.
    for (BinaryBasicBlock *BB : FF) {
      for (auto It = BB->begin(); It != BB->end(); ++It) {
        MCInst &Inst = *It;
        if (BC.MIB->isCFI(Inst))
          continue;
        std::optional<bool> RAState = BC.MIB->getRAState(Inst);
        if (!RAState.has_value()) {
          BC.errs() << "BOLT-ERROR: unknown RAState after inferUnknownStates "
                    << " in function " << BF.getPrintName() << "\n";
          PassFailed = true;
          return;
        }
        if (!FirstIter) {
          // Consecutive instructions with different RAState means we need to
          // add a OpNegateRAState.
          if (*RAState != PrevRAState)
            It = BF.addCFIInstruction(
                BB, It, MCCFIInstruction::createNegateRAState(nullptr));
        } else {
          FirstIter = false;
        }
        PrevRAState = *RAState;
      }
    }
  }
}

void PointerAuthCFIFixup::inferUnknownStates(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();

  // Fill in missing RAStates in simple cases (inside BBs).
  for (BinaryBasicBlock &BB : BF) {
    fillUnknownStateInBB(BC, BB);
  }
  // BasicBlocks which are made entirely of "new instructions" (instructions
  // without RAState annotation) are stubs, and do not have correct unwind info.
  // We should iterate in layout order and fill them based on previous known
  // RAState.
  fillUnknownStubs(BF);
}

void PointerAuthCFIFixup::coverFunctionFragmentStart(BinaryFunction &BF,
                                                     FunctionFragment &FF) {
  BinaryContext &BC = BF.getBinaryContext();
  if (FF.empty())
    return;
  // Find the first BB in the FF which has Instructions.
  // BOLT can generate empty BBs at function splitting which are only used as
  // target labels. We should add the negate-ra-state CFI to the first
  // non-empty BB.
  auto *FirstNonEmpty =
      std::find_if(FF.begin(), FF.end(), [](BinaryBasicBlock *BB) {
        // getFirstNonPseudo returns BB.end() if it does not find any
        // Instructions.
        return BB->getFirstNonPseudo() != BB->end();
      });
  // If a function is already split in the input, the first FF can also start
  // with Signed state. This covers that scenario as well.
  auto II = (*FirstNonEmpty)->getFirstNonPseudo();
  std::optional<bool> RAState = BC.MIB->getRAState(*II);
  if (!RAState.has_value()) {
    BC.errs() << "BOLT-ERROR: unknown RAState after inferUnknownStates "
              << " in function " << BF.getPrintName() << "\n";
    PassFailed = true;
    return;
  }
  if (*RAState)
    BF.addCFIInstruction(*FirstNonEmpty, II,
                         MCCFIInstruction::createNegateRAState(nullptr));
}

std::optional<bool>
PointerAuthCFIFixup::getFirstKnownRAState(BinaryContext &BC,
                                          BinaryBasicBlock &BB) {
  for (const MCInst &Inst : BB) {
    if (BC.MIB->isCFI(Inst))
      continue;
    std::optional<bool> RAState = BC.MIB->getRAState(Inst);
    if (RAState.has_value())
      return RAState;
  }
  return std::nullopt;
}

bool PointerAuthCFIFixup::isUnknownBlock(BinaryContext &BC,
                                         BinaryBasicBlock &BB) {
  std::optional<bool> FirstRAState = getFirstKnownRAState(BC, BB);
  return !FirstRAState.has_value();
}

void PointerAuthCFIFixup::fillUnknownStateInBB(BinaryContext &BC,
                                               BinaryBasicBlock &BB) {

  auto First = BB.getFirstNonPseudo();
  if (First == BB.end())
    return;
  // If the first instruction has unknown RAState, we should copy the first
  // known RAState.
  std::optional<bool> RAState = BC.MIB->getRAState(*First);
  if (!RAState.has_value()) {
    std::optional<bool> FirstRAState = getFirstKnownRAState(BC, BB);
    if (!FirstRAState.has_value())
      // We fill unknown BBs later.
      return;

    BC.MIB->setRAState(*First, *FirstRAState);
  }

  // At this point we know the RAState of the first instruction,
  // so we can propagate the RAStates to all subsequent unknown instructions.
  MCInst Prev = *First;
  for (auto It = First + 1; It != BB.end(); ++It) {
    MCInst &Inst = *It;
    if (BC.MIB->isCFI(Inst))
      continue;

    // No need to check for nullopt: we only entered this loop after the first
    // instruction had its RAState set, and RAState is always set for the
    // previous instruction in the previous iteration of the loop.
    std::optional<bool> PrevRAState = BC.MIB->getRAState(Prev);

    std::optional<bool> RAState = BC.MIB->getRAState(Inst);
    if (!RAState.has_value()) {
      if (BC.MIB->isPSignOnLR(Prev))
        PrevRAState = true;
      else if (BC.MIB->isPAuthOnLR(Prev))
        PrevRAState = false;
      BC.MIB->setRAState(Inst, *PrevRAState);
    }
    Prev = Inst;
  }
}

void PointerAuthCFIFixup::markUnknownBlock(BinaryContext &BC,
                                           BinaryBasicBlock &BB, bool State) {
  // If we call this when an Instruction has either kRASigned or kRAUnsigned
  // annotation, setRAState would fail.
  assert(isUnknownBlock(BC, BB) &&
         "markUnknownBlock should only be called on unknown blocks");
  for (MCInst &Inst : BB) {
    if (BC.MIB->isCFI(Inst))
      continue;
    BC.MIB->setRAState(Inst, State);
  }
}

void PointerAuthCFIFixup::fillUnknownStubs(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();
  bool FirstIter = true;
  MCInst PrevInst;
  for (FunctionFragment &FF : BF.getLayout().fragments()) {
    for (BinaryBasicBlock *BB : FF) {
      if (FirstIter) {
        FirstIter = false;
        if (isUnknownBlock(BC, *BB))
          // If the first BasicBlock is unknown, the function's entry RAState
          // should be used.
          markUnknownBlock(BC, *BB, BF.getInitialRAState());
      } else if (isUnknownBlock(BC, *BB)) {
        // As explained in issue #160989, the unwind info is incorrect for
        // stubs. Indicating the correct RAState without the rest of the unwind
        // info being correct is not useful. Instead, we copy the RAState from
        // the previous instruction.
        std::optional<bool> PrevRAState = BC.MIB->getRAState(PrevInst);
        if (!PrevRAState.has_value()) {
          // No non-cfi instruction encountered in the function yet.
          // This means the RAState is the same as at the function entry.
          markUnknownBlock(BC, *BB, BF.getInitialRAState());
          continue;
        }

        if (BC.MIB->isPSignOnLR(PrevInst))
          PrevRAState = true;
        else if (BC.MIB->isPAuthOnLR(PrevInst))
          PrevRAState = false;
        markUnknownBlock(BC, *BB, *PrevRAState);
      }
      // This function iterates on BasicBlocks, so the PrevInst has to be
      // updated to the last instruction of the current BasicBlock. If the
      // BasicBlock is empty, or only has PseudoInstructions, PrevInst will not
      // be updated.
      auto Last = BB->getLastNonPseudo();
      if (Last != BB->rend())
        PrevInst = *Last;
    }
  }
}

Error PointerAuthCFIFixup::runOnFunctions(BinaryContext &BC) {
  std::atomic<uint64_t> FunctionsModified{0};
  ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
    FunctionsModified++;
    runOnFunction(BF);
  };

  ParallelUtilities::PredicateTy SkipPredicate = [&](const BinaryFunction &BF) {
    // We can skip functions which did not include negate-ra-state CFIs. This
    // includes code using pac-ret hardening as well, if the binary is
    // compiled with `-fno-exceptions -fno-unwind-tables
    // -fno-asynchronous-unwind-tables`
    return !BF.containedNegateRAState() || BF.isIgnored();
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_INST_LINEAR, WorkFun,
      SkipPredicate, "PointerAuthCFIFixup");

  BC.outs() << "BOLT-INFO: rewritten pac-ret DWARF info in "
            << FunctionsModified << " out of " << BC.getBinaryFunctions().size()
            << " functions "
            << format("(%.2lf%%).\n", (100.0 * FunctionsModified) /
                                          BC.getBinaryFunctions().size());
  if (PassFailed)
    return createFatalBOLTError("");
  return Error::success();
}

} // end namespace bolt
} // end namespace llvm

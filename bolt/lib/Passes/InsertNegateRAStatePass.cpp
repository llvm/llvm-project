//===- bolt/Passes/InsertNegateRAStatePass.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the InsertNegateRAStatePass class. It inserts
// OpNegateRAState CFIs to places where the state of two consecutive
// instructions are different.
//
//===----------------------------------------------------------------------===//
#include "bolt/Passes/InsertNegateRAStatePass.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/ParallelUtilities.h"
#include <cstdlib>

using namespace llvm;

namespace llvm {
namespace bolt {

static bool PassFailed = false;

void InsertNegateRAState::runOnFunction(BinaryFunction &BF) {
  if (PassFailed)
    return;

  BinaryContext &BC = BF.getBinaryContext();

  if (BF.getState() == BinaryFunction::State::Empty)
    return;

  if (BF.getState() != BinaryFunction::State::CFG &&
      BF.getState() != BinaryFunction::State::CFG_Finalized) {
    BC.outs() << "BOLT-INFO: no CFG for " << BF.getPrintName()
              << " in InsertNegateRAStatePass\n";
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
        auto RAState = BC.MIB->getRAState(Inst);
        if (!RAState) {
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

void InsertNegateRAState::inferUnknownStates(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();

  // Fill in missing RAStates in simple cases (inside BBs).
  for (BinaryBasicBlock &BB : BF) {
    fillUnknownStateInBB(BC, BB);
  }
  // Some stubs have no predecessors. For those, we iterate once in the layout
  // order to fill their RAState.
  fillUnknownStubs(BF);

  fillUnknownBlocksInCFG(BF);
}

void InsertNegateRAState::coverFunctionFragmentStart(BinaryFunction &BF,
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
  auto RAState = BC.MIB->getRAState(*II);
  if (!RAState) {
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
InsertNegateRAState::getFirstKnownRAState(BinaryContext &BC,
                                          BinaryBasicBlock &BB) {
  for (const MCInst &Inst : BB) {
    if (BC.MIB->isCFI(Inst))
      continue;
    auto RAStateOpt = BC.MIB->getRAState(Inst);
    if (RAStateOpt)
      return RAStateOpt;
  }
  return std::nullopt;
}

void InsertNegateRAState::fillUnknownStateInBB(BinaryContext &BC,
                                               BinaryBasicBlock &BB) {

  auto First = BB.getFirstNonPseudo();
  if (First == BB.end())
    return;
  // If the first instruction has unknown RAState, we should copy the first
  // known RAState.
  auto RAStateOpt = BC.MIB->getRAState(*First);
  if (!RAStateOpt) {
    auto FirstRAState = getFirstKnownRAState(BC, BB);
    if (!FirstRAState)
      // We fill unknown BBs later.
      return;

    BC.MIB->setRAState(*First, *FirstRAState);
  }

  // At this point we know the RAState of the first instruction,
  // so we can propagate the RAStates to all subsequent unknown instructions.
  MCInst Prev = *First;
  for (auto It = BB.begin() + 1; It != BB.end(); ++It) {
    MCInst &Inst = *It;
    if (BC.MIB->isCFI(Inst))
      continue;

    auto PrevRAState = BC.MIB->getRAState(Prev);
    if (!PrevRAState)
      llvm_unreachable("Previous Instruction has no RAState.");

    auto RAState = BC.MIB->getRAState(Inst);
    if (!RAState) {
      if (BC.MIB->isPSignOnLR(Prev))
        PrevRAState = true;
      else if (BC.MIB->isPAuthOnLR(Prev))
        PrevRAState = false;
      BC.MIB->setRAState(Inst, *PrevRAState);
    }
    Prev = Inst;
  }
}

bool InsertNegateRAState::isUnknownBlock(BinaryContext &BC,
                                         BinaryBasicBlock &BB) {
  for (const MCInst &Inst : BB) {
    if (BC.MIB->isCFI(Inst))
      continue;
    auto RAState = BC.MIB->getRAState(Inst);
    if (RAState)
      return false;
  }
  return true;
}

void InsertNegateRAState::markUnknownBlock(BinaryContext &BC,
                                           BinaryBasicBlock &BB, bool State) {
  // If we call this when an Instruction has either kRASigned or kRAUnsigned
  // annotation, setRASigned or setRAUnsigned would fail.
  assert(isUnknownBlock(BC, BB) &&
         "markUnknownBlock should only be called on unknown blocks");
  for (MCInst &Inst : BB) {
    if (BC.MIB->isCFI(Inst))
      continue;
    BC.MIB->setRAState(Inst, State);
  }
}

std::optional<bool> InsertNegateRAState::getRAStateByCFG(BinaryBasicBlock &BB,
                                                         BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();

  auto checkRAState = [&](std::optional<bool> &NeighborRAState, MCInst &Inst) {
    auto RAState = BC.MIB->getRAState(Inst);
    if (!RAState)
      return;
    if (!NeighborRAState) {
      NeighborRAState = *RAState;
      return;
    }
    if (NeighborRAState != *RAState) {
      BC.outs() << "BOLT-WARNING: Conflicting RAState found in function "
                << BF.getPrintName() << ". Function will not be optimized.\n";
      BF.setIgnored();
    }
  };

  // Holds the first found RAState from CFG neighbors.
  std::optional<bool> NeighborRAState = std::nullopt;
  if (BB.pred_size() != 0) {
    for (BinaryBasicBlock *PredBB : BB.predecessors()) {
      //  find last inst of Predecessor with known RA State.
      auto LI = PredBB->getLastNonPseudo();
      if (LI == PredBB->rend())
        continue;
      MCInst &LastInst = *LI;
      checkRAState(NeighborRAState, LastInst);
    }
  } else if (BB.succ_size() != 0) {
    for (BinaryBasicBlock *SuccBB : BB.successors()) {
      //  find first inst of Successor with known RA State.
      auto FI = SuccBB->getFirstNonPseudo();
      if (FI == SuccBB->end())
        continue;
      MCInst &FirstInst = *FI;
      checkRAState(NeighborRAState, FirstInst);
    }
  } else {
    llvm_unreachable("Called getRAStateByCFG on a BB with no preds or succs.");
  }

  return NeighborRAState;
}

void InsertNegateRAState::fillUnknownStubs(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();
  bool FirstIter = true;
  MCInst PrevInst;
  for (FunctionFragment &FF : BF.getLayout().fragments()) {
    for (BinaryBasicBlock *BB : FF) {
      if (!FirstIter && isUnknownBlock(BC, *BB)) {
        // If we have no predecessors or successors, current BB is a Stub called
        // from another BinaryFunction. As of #160989, we have to copy the
        // PrevInst's RAState, because CFIs are already incorrect here.
        if (BB->pred_size() == 0 && BB->succ_size() == 0) {
          auto PrevRAState = BC.MIB->getRAState(PrevInst);
          if (!PrevRAState) {
            llvm_unreachable(
                "Previous Instruction has no RAState in fillUnknownStubs.");
            continue;
          }

          if (BC.MIB->isPSignOnLR(PrevInst)) {
            PrevRAState = true;
          } else if (BC.MIB->isPAuthOnLR(PrevInst)) {
            PrevRAState = false;
          }
          markUnknownBlock(BC, *BB, *PrevRAState);
        }
      }
      if (FirstIter) {
        FirstIter = false;
        if (isUnknownBlock(BC, *BB))
          markUnknownBlock(BC, *BB, false);
      }
      auto Last = BB->getLastNonPseudo();
      if (Last != BB->rend())
        PrevInst = *Last;
    }
  }
}
void InsertNegateRAState::fillUnknownBlocksInCFG(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();

  auto fillUnknowns = [&](BinaryFunction &BF) -> std::pair<int, bool> {
    int Unknowns = 0;
    bool Updated = false;
    for (BinaryBasicBlock &BB : BF) {
      // Only try to iterate if the BB has either predecessors or successors.
      if (isUnknownBlock(BC, BB) &&
          (BB.pred_size() != 0 || BB.succ_size() != 0)) {
        auto RAStateOpt = getRAStateByCFG(BB, BF);
        if (RAStateOpt) {
          markUnknownBlock(BC, BB, *RAStateOpt);
          Updated = true;
        } else {
          Unknowns++;
        }
      }
    }
    return std::pair<int, bool>{Unknowns, Updated};
  };

  while (true) {
    std::pair<int, bool> Iter = fillUnknowns(BF);
    if (Iter.first == 0)
      break;
    if (!Iter.second) {
      BC.errs() << "BOLT-WARNING: Could not infer RAState for " << Iter.first
                << " BBs in function " << BF.getPrintName()
                << ". Function will not be optimized.\n";
      BF.setIgnored();
      break;
    }
  }
}

Error InsertNegateRAState::runOnFunctions(BinaryContext &BC) {
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
      SkipPredicate, "InsertNegateRAStatePass");

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

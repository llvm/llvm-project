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

void InsertNegateRAState::runOnFunction(BinaryFunction &BF) {
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
    MCInst PrevInst;
    // As this pass runs after function splitting, we should only check
    // consecutive instructions inside FunctionFragments.
    for (BinaryBasicBlock *BB : FF) {
      for (auto It = BB->begin(); It != BB->end(); ++It) {
        MCInst &Inst = *It;
        if (BC.MIB->isCFI(Inst))
          continue;
        if (!FirstIter) {
          // Consecutive instructions with different RAState means we need to
          // add a OpNegateRAState.
          if ((BC.MIB->isRASigned(PrevInst) && BC.MIB->isRAUnsigned(Inst)) ||
              (BC.MIB->isRAUnsigned(PrevInst) && BC.MIB->isRASigned(Inst))) {
            It = BF.addCFIInstruction(
                BB, It, MCCFIInstruction::createNegateRAState(nullptr));
          }
        } else {
          FirstIter = false;
        }
        PrevInst = *It;
      }
    }
  }
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
  if (BC.MIB->isRASigned(*((*FirstNonEmpty)->begin()))) {
    BF.addCFIInstruction(*FirstNonEmpty, (*FirstNonEmpty)->begin(),
                         MCCFIInstruction::createNegateRAState(nullptr));
  }
}

void InsertNegateRAState::inferUnknownStates(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();
  bool FirstIter = true;
  MCInst PrevInst;
  for (BinaryBasicBlock &BB : BF) {
    for (MCInst &Inst : BB) {
      if (BC.MIB->isCFI(Inst))
        continue;

      if (!FirstIter && BC.MIB->isRAStateUnknown(Inst)) {
        if (BC.MIB->isRASigned(PrevInst) || BC.MIB->isPSignOnLR(PrevInst)) {
          BC.MIB->setRASigned(Inst);
        } else if (BC.MIB->isRAUnsigned(PrevInst) ||
                   BC.MIB->isPAuthOnLR(PrevInst)) {
          BC.MIB->setRAUnsigned(Inst);
        }
      } else {
        FirstIter = false;
      }
      PrevInst = Inst;
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
  return Error::success();
}

} // end namespace bolt
} // end namespace llvm

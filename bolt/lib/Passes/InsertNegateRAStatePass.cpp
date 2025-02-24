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
#include "bolt/Utils/CommandLineOpts.h"
#include <cstdlib>
#include <fstream>
#include <iterator>

using namespace llvm;

namespace llvm {
namespace bolt {

void InsertNegateRAState::runOnFunction(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();

  if (BF.getState() == BinaryFunction::State::Empty)
    return;

  if (BF.getState() != BinaryFunction::State::CFG &&
      BF.getState() != BinaryFunction::State::CFG_Finalized) {
    BC.outs() << "BOLT-INFO: No CFG for " << BF.getPrintName()
              << " in InsertNegateRAStatePass\n";
    return;
  }

  // If none is inserted, the function doesn't need more work.
  if (!addNegateRAStateAfterPacOrAuth(BF))
    return;

  fixUnknownStates(BF);

  bool FirstIter = true;
  MCInst PrevInst;
  BinaryBasicBlock *PrevBB = nullptr;
  auto *Begin = BF.getLayout().block_begin();
  auto *End = BF.getLayout().block_end();
  for (auto *BB = Begin; BB != End; BB++) {

    // Support for function splitting:
    // if two consecutive BBs are going to end up in different functions,
    // we have to negate the RA State, so the new function starts with a Signed
    // state.
    if (PrevBB != nullptr &&
        PrevBB->getFragmentNum() != (*BB)->getFragmentNum() &&
        BC.MIB->isRASigned(*((*BB)->begin()))) {
      BF.addCFIInstruction(*BB, (*BB)->begin(),
                           MCCFIInstruction::createNegateRAState(nullptr));
    }

    for (auto It = (*BB)->begin(); It != (*BB)->end(); ++It) {

      MCInst &Inst = *It;
      if (BC.MIB->isCFI(Inst))
        continue;

      if (!FirstIter) {
        if ((BC.MIB->isRASigned(PrevInst) && BC.MIB->isRAUnsigned(Inst)) ||
            (BC.MIB->isRAUnsigned(PrevInst) && BC.MIB->isRASigned(Inst))) {

          It = BF.addCFIInstruction(
              *BB, It, MCCFIInstruction::createNegateRAState(nullptr));
        }

      } else {
        FirstIter = false;
      }
      PrevInst = Inst;
    }
    PrevBB = *BB;
  }
}

bool InsertNegateRAState::addNegateRAStateAfterPacOrAuth(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();
  bool FoundAny = false;
  for (BinaryBasicBlock &BB : BF) {
    for (auto Iter = BB.begin(); Iter != BB.end(); ++Iter) {
      MCInst &Inst = *Iter;
      if (BC.MIB->isPSign(Inst) || BC.MIB->isPAuth(Inst)) {
        Iter = BF.addCFIInstruction(
            &BB, Iter + 1, MCCFIInstruction::createNegateRAState(nullptr));
        FoundAny = true;
      }
    }
  }
  return FoundAny;
}

void InsertNegateRAState::fixUnknownStates(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();
  bool FirstIter = true;
  MCInst PrevInst;
  for (BinaryBasicBlock &BB : BF) {
    for (auto It = BB.begin(); It != BB.end(); ++It) {

      MCInst &Inst = *It;
      if (BC.MIB->isCFI(Inst))
        continue;

      if (!FirstIter && BC.MIB->isRAStateUnknown(Inst)) {
        if (BC.MIB->isRASigned(PrevInst) || BC.MIB->isRASigning(PrevInst)) {
          BC.MIB->setRASigned(Inst);
        } else if (BC.MIB->isRAUnsigned(PrevInst) ||
                   BC.MIB->isAuthenticating(PrevInst)) {
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
  ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
    runOnFunction(BF);
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_TRIVIAL, WorkFun, nullptr,
      "InsertNegateRAStatePass");

  return Error::success();
}

} // end namespace bolt
} // end namespace llvm

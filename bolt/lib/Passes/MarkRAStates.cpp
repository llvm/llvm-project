//===- bolt/Passes/MarkRAStates.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MarkRAStates class.
// Three CFIs have an influence on the RA State of an instruction:
// - NegateRAState flips the RA State,
// - RememberState pushes the RA State to a stack,
// - RestoreState pops the RA State from the stack.
// These are saved as MCAnnotations on instructions they refer to at CFI
// reading (in CFIReaderWriter::fillCFIInfoFor). In this pass, we can work out
// the RA State of each instruction, and save it as new MCAnnotations. The new
// annotations are Signing, Signed, Authenticating and Unsigned. After
// optimizations, .cfi_negate_ra_state CFIs are added to the places where the
// state changes in InsertNegateRAStatePass.
//
//===----------------------------------------------------------------------===//
#include "bolt/Passes/MarkRAStates.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Utils/CommandLineOpts.h"
#include <cstdlib>
#include <fstream>
#include <iterator>

#include <iostream>
#include <optional>
#include <stack>

using namespace llvm;

namespace llvm {
namespace bolt {

void MarkRAStates::runOnFunction(BinaryFunction &BF) {

  if (BF.isIgnored())
    return;

  BinaryContext &BC = BF.getBinaryContext();

  for (BinaryBasicBlock &BB : BF) {
    for (auto It = BB.begin(); It != BB.end(); ++It) {
      MCInst &Inst = *It;
      if ((BC.MIB->isPSign(Inst) || BC.MIB->isPAuth(Inst)) &&
          !BC.MIB->hasNegateRAState(Inst)) {
        // no .cfi_negate_ra_state attached to signing or authenticating instr
        // means, that this is a function with handwritten assembly, which might
        // not respect Clang's conventions (e.g. tailcalls are always
        // authenticated, so functions always start with unsigned RAState when
        // working with compiler-generated code)
        BF.setIgnored();
        BC.outs() << "BOLT-INFO: ignoring RAStates in function "
                  << BF.getPrintName() << "\n";
        return;
      }
    }
  }

  bool RAState = false;
  std::stack<bool> RAStateStack;

  for (BinaryBasicBlock &BB : BF) {
    for (auto It = BB.begin(); It != BB.end(); ++It) {

      MCInst &Inst = *It;
      if (BC.MIB->isCFI(Inst))
        continue;

      if (BC.MIB->isPSign(Inst)) {
        assert(!RAState && "Signed RA State before PSign");
        BC.MIB->setRASigning(Inst);

      } else if (BC.MIB->isPAuth(Inst)) {
        assert(RAState && "Unsigned RA State before PAuth");
        BC.MIB->setAuthenticating(Inst);
      } else if (RAState) {
        BC.MIB->setRASigned(Inst);
      } else {
        BC.MIB->setRAUnsigned(Inst);
      }

      // Updating RAState. All updates are valid from the next instruction.
      // Because the same instruction can have remember and restore, the order
      // here is relevant. This is the reason to loop over Annotations instead
      // of just checking each in a predefined order.
      for (unsigned int Idx = 0; Idx < Inst.getNumOperands(); Idx++) {
        std::optional<int64_t> Annotation =
            BC.MIB->getAnnotationAtOpIndex(Inst, Idx);
        if (!Annotation)
          continue;
        if (Annotation == MCPlus::MCAnnotation::kNegateState)
          RAState = !RAState;
        else if (Annotation == MCPlus::MCAnnotation::kRememberState)
          RAStateStack.push(RAState);
        else if (Annotation == MCPlus::MCAnnotation::kRestoreState) {
          RAState = RAStateStack.top();
          RAStateStack.pop();
        }
      }
    }
  }
}

Error MarkRAStates::runOnFunctions(BinaryContext &BC) {
  ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
    runOnFunction(BF);
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_TRIVIAL, WorkFun, nullptr,
      "MarkRAStates");

  return Error::success();
}

} // end namespace bolt
} // end namespace llvm

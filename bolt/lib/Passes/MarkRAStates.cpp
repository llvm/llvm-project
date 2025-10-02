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
#include <cstdlib>
#include <optional>
#include <stack>

using namespace llvm;

namespace llvm {
namespace bolt {

bool MarkRAStates::runOnFunction(BinaryFunction &BF) {

  BinaryContext &BC = BF.getBinaryContext();

  for (const BinaryBasicBlock &BB : BF) {
    for (const MCInst &Inst : BB) {
      if ((BC.MIB->isPSignOnLR(Inst) ||
           (BC.MIB->isPAuthOnLR(Inst) && !BC.MIB->isPAuthAndRet(Inst))) &&
          !BC.MIB->hasNegateRAState(Inst)) {
        // Not all functions have .cfi_negate_ra_state in them. But if one does,
        // we expect psign/pauth instructions to have the hasNegateRAState
        // annotation.
        BF.setIgnored();
        BC.outs() << "BOLT-INFO: inconsistent RAStates in function "
                  << BF.getPrintName()
                  << ": ptr sign/auth inst without .cfi_negate_ra_state\n";
        return false;
      }
    }
  }

  bool RAState = BF.getInitialRAState();
  std::stack<bool> RAStateStack;
  RAStateStack.push(RAState);

  for (BinaryBasicBlock &BB : BF) {
    for (MCInst &Inst : BB) {
      if (BC.MIB->isCFI(Inst))
        continue;

      if (BC.MIB->isPSignOnLR(Inst)) {
        if (RAState) {
          // RA signing instructions should only follow unsigned RA state.
          BC.outs() << "BOLT-INFO: inconsistent RAStates in function "
                    << BF.getPrintName()
                    << ": ptr signing inst encountered in Signed RA state\n";
          BF.setIgnored();
          return false;
        }
        // The signing instruction itself is unsigned, the next will be
        // signed.
        BC.MIB->setRAUnsigned(Inst);
      } else if (BC.MIB->isPAuthOnLR(Inst)) {
        if (!RAState) {
          // RA authenticating instructions should only follow signed RA state.
          BC.outs() << "BOLT-INFO: inconsistent RAStates in function "
                    << BF.getPrintName()
                    << ": ptr authenticating inst encountered in Unsigned RA "
                       "state\n";
          BF.setIgnored();
          return false;
        }
        // The authenticating instruction itself is signed, but the next will be
        // unsigned.
        BC.MIB->setRASigned(Inst);
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
  return true;
}

Error MarkRAStates::runOnFunctions(BinaryContext &BC) {
  std::atomic<uint64_t> FunctionsIgnored{0};
  ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
    if (!runOnFunction(BF)) {
      FunctionsIgnored++;
    }
  };

  ParallelUtilities::PredicateTy SkipPredicate = [&](const BinaryFunction &BF) {
    // We can skip functions which did not include negate-ra-state CFIs. This
    // includes code using pac-ret hardening as well, if the binary is
    // compiled with `-fno-exceptions -fno-unwind-tables
    // -fno-asynchronous-unwind-tables`
    return !BF.containedNegateRAState() || BF.isIgnored();
  };

  int Total = llvm::count_if(
      BC.getBinaryFunctions(),
      [&](std::pair<const unsigned long, BinaryFunction> &P) {
        return P.second.containedNegateRAState() && !P.second.isIgnored();
      });

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_INST_LINEAR, WorkFun,
      SkipPredicate, "MarkRAStates");
  BC.outs() << "BOLT-INFO: MarkRAStates ran on " << Total
            << " functions. Ignored " << FunctionsIgnored << " functions "
            << format("(%.2lf%%)", (100.0 * FunctionsIgnored) / Total)
            << " because of CFI inconsistencies\n";

  return Error::success();
}

} // end namespace bolt
} // end namespace llvm

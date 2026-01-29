//===- bolt/Passes/PointerAuthCFIAnalyzer.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the PointerAuthCFIAnalyzer class.
// Three CFIs have an influence on the RA State of an instruction:
// - NegateRAState flips the RA State,
// - RememberState pushes the RA State to a stack,
// - RestoreState pops the RA State from the stack.
// These are saved as MCAnnotations on instructions they refer to at CFI
// reading (in CFIReaderWriter::fillCFIInfoFor). In this pass, we can work out
// the RA State of each instruction, and save it as new MCAnnotations. The new
// annotations are Signing, Signed, Authenticating and Unsigned. After
// optimizations, .cfi_negate_ra_state CFIs are added to the places where the
// state changes in PointerAuthCFIFixup.
//
//===----------------------------------------------------------------------===//
#include "bolt/Passes/PointerAuthCFIAnalyzer.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/ParallelUtilities.h"
#include <cstdlib>
#include <optional>
#include <stack>

using namespace llvm;

namespace opts {
extern llvm::cl::opt<unsigned> Verbosity;
} // namespace opts

namespace llvm {
namespace bolt {

bool PointerAuthCFIAnalyzer::runOnFunction(BinaryFunction &BF) {

  BinaryContext &BC = BF.getBinaryContext();

  for (const BinaryBasicBlock &BB : BF) {
    for (const MCInst &Inst : BB) {
      if ((BC.MIB->isPSignOnLR(Inst) ||
           (BC.MIB->isPAuthOnLR(Inst) && !BC.MIB->isPAuthAndRet(Inst))) &&
          !BC.MIB->hasNegateRAState(Inst)) {
        // Not all functions have .cfi_negate_ra_state in them. But if one does,
        // we expect psign/pauth instructions to have the hasNegateRAState
        // annotation.
        if (opts::Verbosity >= 1)
          BC.outs() << "BOLT-INFO: inconsistent RAStates in function "
                    << BF.getPrintName()
                    << ": ptr sign/auth inst without .cfi_negate_ra_state\n";
        std::lock_guard<std::mutex> Lock(IgnoreMutex);
        BF.setIgnored();
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
          if (opts::Verbosity >= 1)
            BC.outs() << "BOLT-INFO: inconsistent RAStates in function "
                      << BF.getPrintName()
                      << ": ptr signing inst encountered in Signed RA state\n";
          std::lock_guard<std::mutex> Lock(IgnoreMutex);
          BF.setIgnored();
          return false;
        }
      } else if (BC.MIB->isPAuthOnLR(Inst)) {
        if (!RAState) {
          // RA authenticating instructions should only follow signed RA state.
          if (opts::Verbosity >= 1)
            BC.outs() << "BOLT-INFO: inconsistent RAStates in function "
                      << BF.getPrintName()
                      << ": ptr authenticating inst encountered in Unsigned RA "
                         "state\n";
          std::lock_guard<std::mutex> Lock(IgnoreMutex);
          BF.setIgnored();
          return false;
        }
      }

      BC.MIB->setRAState(Inst, RAState);

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

Error PointerAuthCFIAnalyzer::runOnFunctions(BinaryContext &BC) {
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

  int Total = llvm::count_if(BC.getBinaryFunctions(), [&](auto &P) {
    return P.second.containedNegateRAState() && !P.second.isIgnored();
  });

  if (Total == 0)
    return Error::success();

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_INST_LINEAR, WorkFun,
      SkipPredicate, "PointerAuthCFIAnalyzer");

  float IgnoredPercent = (100.0 * FunctionsIgnored) / Total;
  BC.outs() << "BOLT-INFO: PointerAuthCFIAnalyzer ran on " << Total
            << " functions. Ignored " << FunctionsIgnored << " functions "
            << format("(%.2lf%%)", IgnoredPercent)
            << " because of CFI inconsistencies\n";

  // Errors in the input are expected from two sources:
  // - compilers emitting incorrect CFIs. This happens more frequently with
  //   older compiler versions, but it should not account for a large
  //   percentage.
  // - input binary is using synchronous unwind tables. This means that after
  //   call sites, the unwind CFIs are dropped: the pass sees missing
  //   .cfi_negate_ra_state from autiasp instructions. If this is the case, a
  //   larger percentage of functions will be ignored.
  //
  // This is why the 10% threshold was chosen: we should not warn about
  // synchronous unwind tables if only a few % are ignored.
  if (IgnoredPercent >= 10.0)
    BC.outs() << "BOLT-WARNING: PointerAuthCFIAnalyzer only supports "
                 "asynchronous unwind tables. For C compilers, see "
                 "-fasynchronous-unwind-tables.\n";

  return Error::success();
}

} // end namespace bolt
} // end namespace llvm

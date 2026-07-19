//===- bolt/Passes/PointerAuthCFIFixup.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the PointerAuthCFIFixup class.
//
//===----------------------------------------------------------------------===//
#ifndef BOLT_PASSES_POINTER_AUTH_CFI_FIXUP
#define BOLT_PASSES_POINTER_AUTH_CFI_FIXUP

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

class PointerAuthCFIFixup : public BinaryFunctionPass {
public:
  explicit PointerAuthCFIFixup(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "pointer-auth-cfi-fixup"; }

  /// Pass entry point
  Error runOnFunctions(BinaryContext &BC) override;
  void runOnFunction(BinaryFunction &BF);

private:
  /// Because states are tracked as MCAnnotations on individual instructions,
  /// newly inserted instructions do not have a state associated with them.
  /// Uses fillUnknownStateInBB and fillUnknownStubs.
  void inferUnknownStates(BinaryFunction &BF);

  /// Simple case: copy RAStates to unknown insts from previous inst.
  /// If the first inst has unknown state, copy set it to the first known state.
  /// Accounts for signing and authenticating insts.
  void fillUnknownStateInBB(BinaryContext &BC, BinaryBasicBlock &BB);

  /// Fill in RAState in BasicBlocks consisting entirely of new instructions.
  /// As of #160989, we have to copy the RAState from the previous BB in the
  /// layout, because CFIs are already incorrect here.
  void fillUnknownStubs(BinaryFunction &BF);

  /// Returns the first known RAState from \p BB, or std::nullopt if all are
  /// unknown.
  std::optional<bool> getFirstKnownRAState(BinaryContext &BC,
                                           BinaryBasicBlock &BB);

  /// \p Return true if all instructions have unknown RAState.
  bool isUnknownBlock(BinaryContext &BC, BinaryBasicBlock &BB);

  /// Set all instructions in \p BB to \p State.
  void markUnknownBlock(BinaryContext &BC, BinaryBasicBlock &BB, bool State);

  /// Support for function splitting:
  /// if two consecutive BBs with Signed state are going to end up in different
  /// functions (so are held by different FunctionFragments), we have to add a
  /// OpNegateRAState to the beginning of the newly split function, so it starts
  /// with a Signed state.
  void coverFunctionFragmentStart(BinaryFunction &BF, FunctionFragment &FF);
};

} // namespace bolt
} // namespace llvm
#endif

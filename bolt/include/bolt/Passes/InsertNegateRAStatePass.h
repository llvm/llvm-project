//===- bolt/Passes/InsertNegateRAStatePass.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the InsertNegateRAStatePass class.
//
//===----------------------------------------------------------------------===//
#ifndef BOLT_PASSES_INSERT_NEGATE_RA_STATE_PASS
#define BOLT_PASSES_INSERT_NEGATE_RA_STATE_PASS

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

class InsertNegateRAState : public BinaryFunctionPass {
public:
  explicit InsertNegateRAState() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "insert-negate-ra-state-pass"; }

  /// Pass entry point
  Error runOnFunctions(BinaryContext &BC) override;
  void runOnFunction(BinaryFunction &BF);

private:
  /// Because states are tracked as MCAnnotations on individual instructions,
  /// newly inserted instructions do not have a state associated with them.
  /// New states are "inherited" from the last known state.
  void inferUnknownStates(BinaryFunction &BF);

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

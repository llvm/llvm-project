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
#include <stack>

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
  /// Loops over all instructions and adds OpNegateRAState CFI
  /// after any pointer signing or authenticating instructions.
  /// Returns true, if any OpNegateRAState CFIs were added.
  bool addNegateRAStateAfterPacOrAuth(BinaryFunction &BF);
  /// Because states are tracked as MCAnnotations on individual instructions,
  /// newly inserted instructions do not have a state associated with them.
  /// New states are "inherited" from the last known state.
  void fixUnknownStates(BinaryFunction &BF);
};

} // namespace bolt
} // namespace llvm
#endif

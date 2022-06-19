//===- bolt/Passes/LoopInversionPass.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_LOOPINVERSION_H
#define BOLT_PASSES_LOOPINVERSION_H

#include "bolt/Passes/BinaryPasses.h"

// This pass founds cases when BBs have layout:
// #BB0:
// ....
// #BB1:
// cmp
// cond_jmp #BB3
// #BB2:
// <loop body>
// jmp #BB1
// #BB3:
// <loop exit>
//
// And swaps BB1 and BB2:
// #BB0:
// ....
// jmp #BB1
// #BB2:
// <loop body>
// #BB1:
// cmp
// cond_njmp #BB2
// #BB3:
// <loop exit>
//
// And vice versa depending on the profile information.
// The advantage is that the loop uses only one conditional jump,
// the unconditional jump is only used once on the loop start.

namespace llvm {
namespace bolt {

class LoopInversionPass : public BinaryFunctionPass {
public:
  explicit LoopInversionPass() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "loop-inversion-opt"; }

  /// Pass entry point
  void runOnFunctions(BinaryContext &BC) override;
  bool runOnFunction(BinaryFunction &Function);
};

} // namespace bolt
} // namespace llvm

#endif

//===- bolt/Passes/JumpTableTrampoline.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Jump table trampolines insertion pass.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_JUMP_TABLE_TRAMPOLINE_H
#define BOLT_PASSES_JUMP_TABLE_TRAMPOLINE_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

/// This pass inserts trampolines for entries in cold fragment into the hot
/// fragment so that offsets fit into the original jump table entry size.
class JumpTableTrampoline : public BinaryFunctionPass {
  DenseSet<const BinaryFunction *> Modified;

  /// Run a pass for \p Function
  void optimizeFunction(BinaryFunction &Function);

public:
  explicit JumpTableTrampoline(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "jump-table-trampoline"; }
  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF) && Modified.count(&BF);
  }
  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif

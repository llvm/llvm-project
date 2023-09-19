//===- bolt/Passes/FixRISCVCallsPass.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the FixRISCVCallsPass class, which replaces all types of
// calls with PseudoCALL pseudo instructions. This ensures that relaxed calls
// get expanded to auipc/jalr pairs so that BOLT can freely reassign function
// addresses without having to worry about the limited range of relaxed calls.
// Using PseudoCALL also ensures that the RISC-V backend inserts the necessary
// relaxation-related relocations to allow JITLink to relax instruction back to
// shorter versions where possible.
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_FIXRISCVCALLSPASS_H
#define BOLT_PASSES_FIXRISCVCALLSPASS_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

class FixRISCVCallsPass : public BinaryFunctionPass {
  void runOnFunction(BinaryFunction &Function);

public:
  explicit FixRISCVCallsPass(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "fix-riscv-calls"; }

  /// Pass entry point
  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif

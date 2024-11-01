//===- bolt/Passes/FixRISCVCallsPass.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the FixRISCVCallsPass class, which sets the JALR immediate
// to 0 for AUIPC/JALR pairs with a R_RISCV_CALL(_PLT) relocation. This is
// necessary since MC expects it to be zero in order to or-in fixups.
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

//===- RISCVLongJmpPass.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_RISCVLONGJMPPASS_H
#define BOLT_PASSES_RISCVLONGJMPPASS_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm::bolt {

/// Insert long jumps between function fragments using dead scratch registers.
/// Keep fragments together when no safe scratch register is available.
class RISCVLongJmpPass : public BinaryFunctionPass {
public:
  explicit RISCVLongJmpPass(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}
  const char *getName() const override { return "riscv-long-jmp"; }
  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace llvm::bolt

#endif

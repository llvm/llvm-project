//===- bolt/Passes/AArch64RelaxationPass.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AArch64RelaxationPass class, which replaces AArch64
// non-local ADR/LDR instructions with ADRP + ADD/LDR due to small offset
// range of ADR and LDR instruction (+- 1MB) which could be easily overflowed
// after BOLT optimizations. Such problems are usually connected with errata
// 843419: https://developer.arm.com/documentation/epm048406/2100/
// The linker could replace ADRP instruction with ADR in some cases.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_AARCH64RELAXATIONPASS_H
#define BOLT_PASSES_AARCH64RELAXATIONPASS_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

class AArch64RelaxationPass : public BinaryFunctionPass {
public:
  explicit AArch64RelaxationPass(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "aarch64-relaxation"; }

  /// Pass entry point
  Error runOnFunctions(BinaryContext &BC) override;
  void runOnFunction(BinaryFunction &BF);
};

} // namespace bolt
} // namespace llvm

#endif

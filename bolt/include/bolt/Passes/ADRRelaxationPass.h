//===- bolt/Passes/ADRRelaxationPass.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ADRRelaxationPass class, which replaces AArch64
// non-local ADR instructions with ADRP + ADD due to small offset range of ADR
// instruction (+- 1MB) which could be easily overflowed after BOLT
// optimizations. Such problems are usually connected with errata 843419
// https://developer.arm.com/documentation/epm048406/2100/
// The linker could replace ADRP instruction with ADR in some cases.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_ADRRELAXATIONPASS_H
#define BOLT_PASSES_ADRRELAXATIONPASS_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

class ADRRelaxationPass : public BinaryFunctionPass {
public:
  explicit ADRRelaxationPass(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "adr-relaxation"; }

  /// Pass entry point
  Error runOnFunctions(BinaryContext &BC) override;
  void runOnFunction(BinaryFunction &BF);
};

} // namespace bolt
} // namespace llvm

#endif

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
  /// Number and total size of constant island copies emitted with non-simple
  /// functions that reference another function's island.
  uint64_t NumIslandCopies{0};
  uint64_t IslandCopyBytes{0};

  /// Point the ADR/LDR Inst in BF at a copy of the constant island it
  /// references in another function. The copy is emitted with BF, so the
  /// reference needs no relaxation. Returns false if this attempt failed.
  bool redirectToIslandCopy(BinaryFunction &BF, MCInst &Inst, unsigned OpNum,
                            const MCSymbol &Symbol);

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

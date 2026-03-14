//===- bolt/Passes/ValidateMemRefs.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_VALIDATEMEMREFS_H
#define BOLT_PASSES_VALIDATEMEMREFS_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm::bolt {

/// Post processing to check for memory references that cause a symbol
/// in data section to be ambiguous, requiring us to avoid moving that
/// object or disambiguating such references. This is currently
/// limited to fixing false references to the location of jump tables.
///
class ValidateMemRefs : public BinaryFunctionPass {
public:
  explicit ValidateMemRefs(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "validate-mem-refs"; }

  Error runOnFunctions(BinaryContext &BC) override;

private:
  bool checkAndFixJTReference(BinaryFunction &BF, MCInst &Inst,
                              uint32_t OperandNum, const MCSymbol *Sym,
                              uint64_t Offset);
  void runOnFunction(BinaryFunction &BF);

  static std::atomic<std::uint64_t> ReplacedReferences;
};

} // namespace llvm::bolt

#endif

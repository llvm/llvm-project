//===- bolt/Passes/AsmDump.h - Dump BinaryFunction as assembly --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BinaryPass to dump BinaryFunction state (CFG, profile data, jump tables,
// CFI state) as assembly.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_ASMDUMP_H
#define BOLT_PASSES_ASMDUMP_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

class AsmDumpPass : public BinaryFunctionPass {
public:
  explicit AsmDumpPass() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "asm-dump"; }

  bool shouldPrint(const BinaryFunction &BF) const override { return false; }

  /// Pass entry point
  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif

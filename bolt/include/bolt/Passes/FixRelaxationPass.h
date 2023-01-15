//===- bolt/Passes/ADRRelaxationPass.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the FixRelaxations class, which locates instructions with
// wrong targets and fixes them. Such problems usually occures when linker
// relaxes (changes) instructions, but doesn't fix relocations types properly
// for them.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_FIXRELAXATIONPASS_H
#define BOLT_PASSES_FIXRELAXATIONPASS_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

class FixRelaxations : public BinaryFunctionPass {
  void runOnFunction(BinaryFunction &Function);

public:
  explicit FixRelaxations(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "fix-relaxations"; }

  /// Pass entry point
  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif

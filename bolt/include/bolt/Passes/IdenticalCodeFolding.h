//===- bolt/Passes/IdenticalCodeFolding.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_IDENTICAL_CODE_FOLDING_H
#define BOLT_PASSES_IDENTICAL_CODE_FOLDING_H

#include "bolt/Core/BinaryFunction.h"
#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

/// An optimization that replaces references to identical functions with
/// references to a single one of them.
///
class IdenticalCodeFolding : public BinaryFunctionPass {
protected:
  bool shouldOptimize(const BinaryFunction &BF) const override {
    if (BF.hasUnknownControlFlow())
      return false;
    if (BF.isFolded())
      return false;
    if (BF.hasSDTMarker())
      return false;
    return BinaryFunctionPass::shouldOptimize(BF);
  }

public:
  explicit IdenticalCodeFolding(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "identical-code-folding"; }
  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif

//===- bolt/Passes/AllocCombiner.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_FRAMEDEFRAG_H
#define BOLT_PASSES_FRAMEDEFRAG_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

class AllocCombinerPass : public BinaryFunctionPass {
  /// Stats aggregating variables
  uint64_t NumCombined{0};
  uint64_t DynamicCountCombined{0};
  DenseSet<const BinaryFunction *> FuncsChanged;

  void combineAdjustments(BinaryFunction &BF);

public:
  explicit AllocCombinerPass(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "alloc-combiner"; }

  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF) && FuncsChanged.count(&BF) > 0;
  }

  /// Pass entry point
  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif

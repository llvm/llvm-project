//===- bolt/Passes/InferNonStale.h - Non-stale profile inference --------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass that runs stale profile matching on functions
// with non-stale/non-inferred profile to improve profile quality.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_INFERNONSTALE_H
#define BOLT_PASSES_INFERNONSTALE_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

/// Run stale profile matching inference on functions with non-stale profile
/// to improve edge count estimates and profile quality.
class InferNonStale : public BinaryFunctionPass {
  void runOnFunction(BinaryFunction &BF);

public:
  explicit InferNonStale(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "infer-non-stale"; }

  /// Pass entry point
  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif

//===- bolt/Passes/PointerAuthCFIAnalyzer.h -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the PointerAuthCFIAnalyzer class.
//
//===----------------------------------------------------------------------===//
#ifndef BOLT_PASSES_POINTER_AUTH_CFI_ANALYZER
#define BOLT_PASSES_POINTER_AUTH_CFI_ANALYZER

#include "bolt/Passes/BinaryPasses.h"
#include <mutex>

namespace llvm {
namespace bolt {

class PointerAuthCFIAnalyzer : public BinaryFunctionPass {
  // setIgnored() is not thread-safe, but the pass is running on functions in
  // parallel.
  std::mutex IgnoreMutex;

public:
  explicit PointerAuthCFIAnalyzer(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "pointer-auth-cfi-analyzer"; }

  /// Pass entry point
  Error runOnFunctions(BinaryContext &BC) override;
  bool runOnFunction(BinaryFunction &BF);
};

} // namespace bolt
} // namespace llvm
#endif

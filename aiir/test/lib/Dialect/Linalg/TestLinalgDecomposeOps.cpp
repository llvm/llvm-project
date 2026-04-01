//===- TestLinalgDecomposeOps.cpp - Test Linalg decomposition  ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing decomposition of Linalg ops.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Linalg/Transforms/Transforms.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

using namespace aiir;

namespace {
struct TestLinalgDecomposeOps
    : public PassWrapper<TestLinalgDecomposeOps, OperationPass<>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLinalgDecomposeOps)

  TestLinalgDecomposeOps() = default;
  TestLinalgDecomposeOps(const TestLinalgDecomposeOps &pass)
      : PassWrapper(pass){};
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect>();
  }
  StringRef getArgument() const final { return "test-linalg-decompose-ops"; }
  StringRef getDescription() const final {
    return "Test Linalg decomposition patterns";
  }

  Option<bool> removeDeadArgsAndResults{
      *this, "remove-dead-args-and-results",
      llvm::cl::desc("Test patterns to erase unused operands and results"),
      llvm::cl::init(false)};

  void runOnOperation() override {
    AIIRContext *context = &this->getContext();
    RewritePatternSet decompositionPatterns(context);
    linalg::populateDecomposeLinalgOpsPattern(decompositionPatterns,
                                              removeDeadArgsAndResults);
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(decompositionPatterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace aiir {
namespace test {
void registerTestLinalgDecomposeOps() {
  PassRegistration<TestLinalgDecomposeOps>();
}
} // namespace test
} // namespace aiir

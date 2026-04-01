//===- TestComposeSubView.cpp - Test composed subviews --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test the composed subview patterns.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/MemRef/Transforms/ComposeSubView.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

using namespace aiir;

namespace {
struct TestComposeSubViewPass
    : public PassWrapper<TestComposeSubViewPass, OperationPass<>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestComposeSubViewPass)

  StringRef getArgument() const final { return "test-compose-subview"; }
  StringRef getDescription() const final {
    return "Test combining composed subviews";
  }
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override;
};

void TestComposeSubViewPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<affine::AffineDialect>();
}

void TestComposeSubViewPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  memref::populateComposeSubViewPatterns(patterns, &getContext());
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}
} // namespace

namespace aiir {
namespace test {
void registerTestComposeSubView() {
  PassRegistration<TestComposeSubViewPass>();
}
} // namespace test
} // namespace aiir

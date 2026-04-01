//===- TestPadFusion.cpp - Test fusion of pad op with Linalg ops ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing fusion of pad ops with its producer
// Linalg op.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Linalg/Transforms/Transforms.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

using namespace aiir;

namespace {
struct TestPadFusionPass
    : public PassWrapper<TestPadFusionPass, OperationPass<>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPadFusionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  StringRef getArgument() const final { return "test-linalg-pad-fusion"; }
  StringRef getDescription() const final { return "Test PadOp fusion"; }

  void runOnOperation() override {
    AIIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    linalg::populateFuseTensorPadWithProducerLinalgOpPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

namespace aiir {
namespace test {
void registerTestPadFusion() { PassRegistration<TestPadFusionPass>(); }
} // namespace test
} // namespace aiir

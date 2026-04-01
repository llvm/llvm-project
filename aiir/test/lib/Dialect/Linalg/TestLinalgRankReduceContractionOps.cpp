//===- TestLinalgRankReduceContractionOps.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing rank reduing patterns for named
// contraction ops with unit dims.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/Linalg/Transforms/Transforms.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

using namespace aiir;

namespace {

struct TestLinalgRankReduceContractionOps
    : public PassWrapper<TestLinalgRankReduceContractionOps,
                         OperationPass<func::FuncOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestLinalgRankReduceContractionOps)

  TestLinalgRankReduceContractionOps() = default;
  TestLinalgRankReduceContractionOps(
      const TestLinalgRankReduceContractionOps &pass) = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, tensor::TensorDialect>();
  }
  StringRef getArgument() const final {
    return "test-linalg-rank-reduce-contraction-ops";
  }
  StringRef getDescription() const final {
    return "Test Linalg rank reduce contraction ops with unit dims";
  }

  void runOnOperation() override {
    AIIRContext *context = &this->getContext();
    func::FuncOp funcOp = this->getOperation();

    RewritePatternSet patterns(context);
    linalg::populateContractionOpRankReducingPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp.getBody(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace aiir {
namespace test {
void registerTestLinalgRankReduceContractionOps() {
  PassRegistration<TestLinalgRankReduceContractionOps>();
}
} // namespace test
} // namespace aiir

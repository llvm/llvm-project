//===-- O0CanonicalizerPass.cpp -- Canonicalization for O0 ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_O0CANONICALIZERPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

// FIR version of Canonicalizer::initialize that excludes cf dialect patterns
// under an option and also provides a way to filter patterns for given
// operations via a callback
void fir::populateCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, bool includeCFPatterns,
    llvm::function_ref<bool(mlir::RegisteredOperationName)> shouldCollect) {
  mlir::MLIRContext *context = patterns.getContext();
  auto includeDialect = [&](mlir::Dialect *dialect) {
    return includeCFPatterns ||
           !mlir::isa<mlir::cf::ControlFlowDialect>(dialect);
  };
  for (mlir::Dialect *dialect : context->getLoadedDialects())
    if (includeDialect(dialect))
      dialect->getCanonicalizationPatterns(patterns);
  for (mlir::RegisteredOperationName op : context->getRegisteredOperations())
    if (includeDialect(&op.getDialect()) &&
        (!shouldCollect || shouldCollect(op)))
      op.getCanonicalizationPatterns(patterns, context);
}

namespace {
class O0CanonicalizerPass
    : public fir::impl::O0CanonicalizerPassBase<O0CanonicalizerPass> {
public:
  mlir::LogicalResult initialize(mlir::MLIRContext *context) override {
    mlir::RewritePatternSet owningPatterns(context);
    fir::populateCanonicalizationPatterns(owningPatterns,
                                          /*includeCFPatterns=*/false);
    patterns = mlir::FrozenRewritePatternSet(std::move(owningPatterns));
    return mlir::success();
  }

  void runOnOperation() override {
    mlir::GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);
    // Like canonicalization, this cleanup is best-effort.
    (void)mlir::applyPatternsGreedily(getOperation(), patterns, config);
  }

private:
  mlir::FrozenRewritePatternSet patterns;
};
} // namespace

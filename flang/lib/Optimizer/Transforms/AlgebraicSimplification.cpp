//===- AlgebraicSimplification.cpp - Simplify algebraic expressions -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines a pass that applies algebraic simplifications
// to operations of Math/Complex/etc. dialects that are used by Flang.
// It is done as a Flang specific pass, because we may want to tune
// the parameters of the patterns for Fortran programs.
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct AlgebraicSimplification
    : public fir::AlgebraicSimplificationBase<AlgebraicSimplification> {
  AlgebraicSimplification(const GreedyRewriteConfig &rewriteConfig) {
    config = rewriteConfig;
  }

  void runOnOperation() override;

  mlir::GreedyRewriteConfig config;
};
} // namespace

void AlgebraicSimplification::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateMathAlgebraicSimplificationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}

std::unique_ptr<mlir::Pass> fir::createAlgebraicSimplificationPass() {
  return std::make_unique<AlgebraicSimplification>(GreedyRewriteConfig());
}

std::unique_ptr<mlir::Pass> fir::createAlgebraicSimplificationPass(
    const mlir::GreedyRewriteConfig &config) {
  return std::make_unique<AlgebraicSimplification>(config);
}

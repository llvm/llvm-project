//====- FlattenCFG.cpp - Flatten CIR CFG ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements pass that inlines CIR operations regions into the parent
// function region.
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace mlir::cir;

namespace {

struct FlattenCFGPass : public FlattenCFGBase<FlattenCFGPass> {

  FlattenCFGPass() = default;
  void runOnOperation() override;
};

void populateFlattenCFGPatterns(RewritePatternSet &patterns) {
  // TODO: add patterns here
}

void FlattenCFGPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateFlattenCFGPatterns(patterns);

  // Collect operations to apply patterns.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
    // TODO: push back operations here
  });

  // Apply patterns.
  if (applyOpPatternsAndFold(ops, std::move(patterns)).failed())
    signalPassFailure();
}

} // namespace

namespace mlir {

std::unique_ptr<Pass> createFlattenCFGPass() {
  return std::make_unique<FlattenCFGPass>();
}

} // namespace mlir

//===- SimplifyRegionLite.cpp -- region simplification lite ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

namespace {

class SimplifyRegionLitePass
    : public fir::SimplifyRegionLiteBase<SimplifyRegionLitePass> {
public:
  void runOnOperation() override;
};

class DummyRewriter : public mlir::PatternRewriter {
public:
  DummyRewriter(mlir::MLIRContext *ctx) : mlir::PatternRewriter(ctx) {}
};

} // namespace

void SimplifyRegionLitePass::runOnOperation() {
  auto op = getOperation();
  auto regions = op->getRegions();
  mlir::RewritePatternSet patterns(op.getContext());
  DummyRewriter rewriter(op.getContext());
  if (regions.empty())
    return;

  (void)mlir::eraseUnreachableBlocks(rewriter, regions);
  (void)mlir::runRegionDCE(rewriter, regions);
}

std::unique_ptr<mlir::Pass> fir::createSimplifyRegionLitePass() {
  return std::make_unique<SimplifyRegionLitePass>();
}

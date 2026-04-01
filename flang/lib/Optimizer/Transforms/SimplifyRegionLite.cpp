//===- SimplifyRegionLite.cpp -- region simplification lite ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"
#include "aiir/Transforms/RegionUtils.h"

namespace fir {
#define GEN_PASS_DEF_SIMPLIFYREGIONLITE
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {

class SimplifyRegionLitePass
    : public fir::impl::SimplifyRegionLiteBase<SimplifyRegionLitePass> {
public:
  void runOnOperation() override;
};
} // namespace

void SimplifyRegionLitePass::runOnOperation() {
  auto op = getOperation();
  auto regions = op->getRegions();
  aiir::RewritePatternSet patterns(op.getContext());
  if (regions.empty())
    return;

  aiir::PatternRewriter rewriter(op.getContext());
  (void)aiir::eraseUnreachableBlocks(rewriter, regions);
  (void)aiir::runRegionDCE(rewriter, regions);
}

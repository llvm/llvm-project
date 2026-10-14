//===- SimplifyRegionLite.cpp -- region simplification lite ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

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
  mlir::Operation *op = getOperation();
  mlir::MutableArrayRef<mlir::Region> regions = op->getRegions();
  if (regions.empty())
    return;

  mlir::PatternRewriter rewriter(op->getContext());
  (void)mlir::eraseUnreachableBlocks(rewriter, regions);
  (void)mlir::runRegionDCE(rewriter, regions);
}

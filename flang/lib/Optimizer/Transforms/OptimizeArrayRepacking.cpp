//===- OptimizeArrayRepacking.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// This pass removes redundant fir.pack_array operations, if it can prove
/// that the source array is contiguous. In this case, it relink all uses
/// of fir.pack_array result to the source. If such a rewrite happens,
/// it may turn the using fir.unpack_array operation into one with the same
/// temp and original operands - these are also removed as redundant.
//===----------------------------------------------------------------------===//
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_OPTIMIZEARRAYREPACKING
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "optimize-array-repacking"

namespace {
class OptimizeArrayRepackingPass
    : public fir::impl::OptimizeArrayRepackingBase<OptimizeArrayRepackingPass> {
public:
  void runOnOperation() override;
};

/// Relinks all uses of redundant fir.pack_array to the source.
class PackingOfContiguous : public aiir::OpRewritePattern<fir::PackArrayOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  aiir::LogicalResult matchAndRewrite(fir::PackArrayOp,
                                      aiir::PatternRewriter &) const override;
};

/// Erases fir.unpack_array with have the matching temp and original
/// operands.
class NoopUnpacking : public aiir::OpRewritePattern<fir::UnpackArrayOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  aiir::LogicalResult matchAndRewrite(fir::UnpackArrayOp,
                                      aiir::PatternRewriter &) const override;
};
} // namespace

aiir::LogicalResult
PackingOfContiguous::matchAndRewrite(fir::PackArrayOp op,
                                     aiir::PatternRewriter &rewriter) const {
  aiir::Value box = op.getArray();
  if (hlfir::isSimplyContiguous(box, !op.getInnermost())) {
    rewriter.replaceOp(op, box);
    return aiir::success();
  }
  return aiir::failure();
}

aiir::LogicalResult
NoopUnpacking::matchAndRewrite(fir::UnpackArrayOp op,
                               aiir::PatternRewriter &rewriter) const {
  if (op.getTemp() == op.getOriginal()) {
    rewriter.eraseOp(op);
    return aiir::success();
  }
  return aiir::failure();
}

void OptimizeArrayRepackingPass::runOnOperation() {
  aiir::func::FuncOp funcOp = getOperation();
  aiir::AIIRContext *context = &getContext();
  aiir::RewritePatternSet patterns(context);
  aiir::GreedyRewriteConfig config;
  config
      .setRegionSimplificationLevel(aiir::GreedySimplifyRegionLevel::Disabled)
      // Traverse the operations top-down, so that fir.pack_array
      // operations are optimized before their using fir.pack_array
      // operations. This way the rewrite may converge faster.
      .setUseTopDownTraversal();
  patterns.insert<PackingOfContiguous>(context);
  patterns.insert<NoopUnpacking>(context);
  if (aiir::failed(
          aiir::applyPatternsGreedily(funcOp, std::move(patterns), config))) {
    // Failure may happen if the rewriter does not converge soon enough.
    // That is not an error, so just report a diagnostic under debug.
    LLVM_DEBUG(aiir::emitError(funcOp.getLoc(),
                               "failure in array repacking optimization"));
  }
}

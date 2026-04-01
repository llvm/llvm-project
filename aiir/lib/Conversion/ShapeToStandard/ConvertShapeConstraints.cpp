//===- ConvertShapeConstraints.cpp - Conversion of shape constraints ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/ShapeToStandard/ShapeToStandard.h"

#include "aiir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/Dialect/Shape/IR/Shape.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace aiir {
#define GEN_PASS_DEF_CONVERTSHAPECONSTRAINTSPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
#include "ShapeToStandard.cpp.inc"
} // namespace

namespace {
class ConvertCstrRequireOp : public OpRewritePattern<shape::CstrRequireOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(shape::CstrRequireOp op,
                                PatternRewriter &rewriter) const override {
    cf::AssertOp::create(rewriter, op.getLoc(), op.getPred(), op.getMsgAttr());
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);
    return success();
  }
};
} // namespace

void aiir::populateConvertShapeConstraintsConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<CstrBroadcastableToRequire>(patterns.getContext());
  patterns.add<CstrEqToRequire>(patterns.getContext());
  patterns.add<ConvertCstrRequireOp>(patterns.getContext());
}

namespace {
// This pass eliminates shape constraints from the program, converting them to
// eager (side-effecting) error handling code. After eager error handling code
// is emitted, witnesses are satisfied, so they are replace with
// `shape.const_witness true`.
class ConvertShapeConstraints
    : public impl::ConvertShapeConstraintsPassBase<ConvertShapeConstraints> {
  void runOnOperation() override {
    auto *func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    populateConvertShapeConstraintsConversionPatterns(patterns);

    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

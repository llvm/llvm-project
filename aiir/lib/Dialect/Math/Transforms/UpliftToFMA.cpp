//===- UpliftToFMA.cpp - Arith to FMA uplifting ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements uplifting from arith ops to math.fma.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Math/IR/Math.h"
#include "aiir/Dialect/Math/Transforms/Passes.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace aiir::math {
#define GEN_PASS_DEF_MATHUPLIFTTOFMA
#include "aiir/Dialect/Math/Transforms/Passes.h.inc"
} // namespace aiir::math

using namespace aiir;

template <typename Op>
static bool isValidForFMA(Op op) {
  return static_cast<bool>(op.getFastmath() & arith::FastMathFlags::contract);
}

namespace {

struct UpliftFma final : OpRewritePattern<arith::AddFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isValidForFMA(op))
      return rewriter.notifyMatchFailure(op, "addf op is not suitable for fma");

    Value c;
    arith::MulFOp ab;
    if ((ab = op.getLhs().getDefiningOp<arith::MulFOp>())) {
      c = op.getRhs();
    } else if ((ab = op.getRhs().getDefiningOp<arith::MulFOp>())) {
      c = op.getLhs();
    } else {
      return rewriter.notifyMatchFailure(op, "no mulf op");
    }

    if (!isValidForFMA(ab))
      return rewriter.notifyMatchFailure(ab, "mulf op is not suitable for fma");

    Value a = ab.getLhs();
    Value b = ab.getRhs();
    arith::FastMathFlags fmf = op.getFastmath() & ab.getFastmath();
    rewriter.replaceOpWithNewOp<math::FmaOp>(op, a, b, c, fmf);
    return success();
  }
};

struct MathUpliftToFMA final
    : math::impl::MathUpliftToFMABase<MathUpliftToFMA> {
  using MathUpliftToFMABase::MathUpliftToFMABase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateUpliftToFMAPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void aiir::populateUpliftToFMAPatterns(RewritePatternSet &patterns) {
  patterns.insert<UpliftFma>(patterns.getContext());
}

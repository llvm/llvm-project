//===- TosaArithConstantToConst.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that converts tensor-valued arith.constant ops
// into tosa.const so that TOSA pipelines operate on a uniform constant form.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Tosa/Transforms/Passes.h"

#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/Quant/IR/QuantTypes.h"
#include "aiir/Dialect/Tosa/IR/TosaOps.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace aiir {
namespace tosa {
#define GEN_PASS_DEF_TOSAARITHCONSTANTTOTOSACONSTPASS
#include "aiir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace aiir

using namespace aiir;
using namespace aiir::tosa;

namespace {

// NOTE: TOSA pipelines already lower their constants through shared Arith
// folding passes, so tensor literals often come back as `arith.constant` even
// after the IR is otherwise TOSA-only. Keep this normalization with the rest of
// the TOSA transforms so any client can re-establish a canonical `tosa.const`
// representation without needing a full Arith->TOSA conversion library.

/// Returns true when `elementType` is natively representable by tosa.const.
static bool isSupportedElementType(Type elementType) {
  if (isa<FloatType>(elementType))
    return true;

  if (auto intType = dyn_cast<IntegerType>(elementType))
    return intType.isSignless() || intType.isUnsigned();

  if (isa<quant::QuantizedType>(elementType))
    return true;

  if (isa<tosa::mxint8Type>(elementType))
    return true;

  return false;
}

class ArithConstantToTosaConst : public OpRewritePattern<arith::ConstantOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constOp,
                                PatternRewriter &rewriter) const override {
    // TOSA constant verification requires a ranked, statically shaped tensor.
    auto resultType = dyn_cast<RankedTensorType>(constOp.getResult().getType());
    if (!resultType || !resultType.hasStaticShape())
      return failure();

    if (!isSupportedElementType(resultType.getElementType()))
      return failure();

    Attribute attr = constOp.getValueAttr();
    auto elementsAttr = dyn_cast<ElementsAttr>(attr);
    if (!elementsAttr)
      return failure();

    auto attrType = dyn_cast<RankedTensorType>(elementsAttr.getType());
    if (!attrType || !attrType.hasStaticShape())
      return failure();
    if (attrType != resultType)
      return failure();

    auto newConst = tosa::ConstOp::create(rewriter, constOp.getLoc(),
                                          resultType, elementsAttr);
    rewriter.replaceOp(constOp, newConst.getResult());
    return success();
  }
};

struct TosaArithConstantToTosaConstPass
    : public tosa::impl::TosaArithConstantToTosaConstPassBase<
          TosaArithConstantToTosaConstPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tosa::TosaDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ArithConstantToTosaConst>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

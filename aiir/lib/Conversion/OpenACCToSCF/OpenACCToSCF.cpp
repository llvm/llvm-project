//===- OpenACCToSCF.cpp - OpenACC condition to SCF if conversion ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/OpenACCToSCF/ConvertOpenACCToSCF.h"

#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/IR/Matchers.h"
#include "aiir/Transforms/DialectConversion.h"

namespace aiir {
#define GEN_PASS_DEF_CONVERTOPENACCTOSCFPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Pattern to transform the `getIfCond` on operation without region into a
/// scf.if and move the operation into the `then` region.
template <typename OpTy>
class ExpandIfCondition : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Early exit if there is no condition.
    if (!op.getIfCond())
      return failure();

    IntegerAttr constAttr;
    if (!matchPattern(op.getIfCond(), m_Constant(&constAttr))) {
      auto ifOp = scf::IfOp::create(rewriter, op.getLoc(), TypeRange(),
                                    op.getIfCond(), false);
      rewriter.modifyOpInPlace(op, [&]() { op.getIfCondMutable().erase(0); });
      auto thenBodyBuilder = ifOp.getThenBodyBuilder(rewriter.getListener());
      thenBodyBuilder.clone(*op.getOperation());
      rewriter.eraseOp(op);
    } else {
      if (constAttr.getInt())
        rewriter.modifyOpInPlace(op, [&]() { op.getIfCondMutable().erase(0); });
      else
        rewriter.eraseOp(op);
    }
    return success();
  }
};
} // namespace

void aiir::populateOpenACCToSCFConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ExpandIfCondition<acc::EnterDataOp>>(patterns.getContext());
  patterns.add<ExpandIfCondition<acc::ExitDataOp>>(patterns.getContext());
  patterns.add<ExpandIfCondition<acc::UpdateOp>>(patterns.getContext());
}

namespace {
struct ConvertOpenACCToSCFPass
    : public impl::ConvertOpenACCToSCFPassBase<ConvertOpenACCToSCFPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertOpenACCToSCFPass::runOnOperation() {
  auto op = getOperation();
  auto *context = op.getContext();

  RewritePatternSet patterns(context);
  ConversionTarget target(*context);
  populateOpenACCToSCFConversionPatterns(patterns);

  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<acc::OpenACCDialect>();

  target.addDynamicallyLegalOp<acc::EnterDataOp>(
      [](acc::EnterDataOp op) { return !op.getIfCond(); });

  target.addDynamicallyLegalOp<acc::ExitDataOp>(
      [](acc::ExitDataOp op) { return !op.getIfCond(); });

  target.addDynamicallyLegalOp<acc::UpdateOp>(
      [](acc::UpdateOp op) { return !op.getIfCond(); });

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

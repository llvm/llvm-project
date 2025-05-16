//===-- FIRToSCF.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

namespace fir {
#define GEN_PASS_DEF_FIRTOSCFPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace mlir;

namespace {
class FIRToSCFPass : public fir::impl::FIRToSCFPassBase<FIRToSCFPass> {
public:
  void runOnOperation() override;
};
} // namespace

struct DoLoopConversion : public OpRewritePattern<fir::DoLoopOp> {
  using OpRewritePattern<fir::DoLoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(fir::DoLoopOp doLoopOp,
                                PatternRewriter &rewriter) const override {
    auto loc = doLoopOp.getLoc();
    bool hasFinalValue = doLoopOp.getFinalValue().has_value();

    // Get loop values from the DoLoopOp
    auto low = doLoopOp.getLowerBound();
    auto high = doLoopOp.getUpperBound();
    assert(low && high && "must be a Value");
    auto step = doLoopOp.getStep();
    llvm::SmallVector<mlir::Value> iterArgs;
    if (hasFinalValue)
      iterArgs.push_back(low);
    iterArgs.append(doLoopOp.getIterOperands().begin(),
                    doLoopOp.getIterOperands().end());

    // Caculate the trip count.
    auto diff = rewriter.create<mlir::arith::SubIOp>(loc, high, low);
    auto distance = rewriter.create<mlir::arith::AddIOp>(loc, diff, step);
    auto tripCount = rewriter.create<mlir::arith::DivSIOp>(loc, distance, step);
    auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto scfForOp =
        rewriter.create<scf::ForOp>(loc, zero, tripCount, one, iterArgs);

    auto &loopOps = doLoopOp.getBody()->getOperations();
    auto resultOp = cast<fir::ResultOp>(doLoopOp.getBody()->getTerminator());
    auto results = resultOp.getOperands();
    Block *loweredBody = scfForOp.getBody();

    loweredBody->getOperations().splice(loweredBody->begin(), loopOps,
                                        loopOps.begin(),
                                        std::prev(loopOps.end()));

    rewriter.setInsertionPointToStart(loweredBody);
    Value iv =
        rewriter.create<arith::MulIOp>(loc, scfForOp.getInductionVar(), step);
    iv = rewriter.create<arith::AddIOp>(loc, low, iv);

    if (!results.empty()) {
      rewriter.setInsertionPointToEnd(loweredBody);
      rewriter.create<scf::YieldOp>(resultOp->getLoc(), results);
    }
    doLoopOp.getInductionVar().replaceAllUsesWith(iv);
    rewriter.replaceAllUsesWith(doLoopOp.getRegionIterArgs(),
                                hasFinalValue
                                    ? scfForOp.getRegionIterArgs().drop_front()
                                    : scfForOp.getRegionIterArgs());

    // Copy loop annotations from the do loop to the loop entry condition.
    if (auto ann = doLoopOp.getLoopAnnotation())
      scfForOp->setAttr("loop_annotation", *ann);

    rewriter.replaceOp(doLoopOp, scfForOp);
    return success();
  }
};

void FIRToSCFPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<DoLoopConversion>(patterns.getContext());
  ConversionTarget target(getContext());
  target.addIllegalOp<fir::DoLoopOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> fir::createFIRToSCFPass() {
  return std::make_unique<FIRToSCFPass>();
}

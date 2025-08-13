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

namespace {
class FIRToSCFPass : public fir::impl::FIRToSCFPassBase<FIRToSCFPass> {
public:
  void runOnOperation() override;
};

struct DoLoopConversion : public mlir::OpRewritePattern<fir::DoLoopOp> {
  using OpRewritePattern<fir::DoLoopOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::DoLoopOp doLoopOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = doLoopOp.getLoc();
    bool hasFinalValue = doLoopOp.getFinalValue().has_value();

    // Get loop values from the DoLoopOp
    mlir::Value low = doLoopOp.getLowerBound();
    mlir::Value high = doLoopOp.getUpperBound();
    assert(low && high && "must be a Value");
    mlir::Value step = doLoopOp.getStep();
    llvm::SmallVector<mlir::Value> iterArgs;
    if (hasFinalValue)
      iterArgs.push_back(low);
    iterArgs.append(doLoopOp.getIterOperands().begin(),
                    doLoopOp.getIterOperands().end());

    // fir.do_loop iterates over the interval [%l, %u], and the step may be
    // negative. But scf.for iterates over the interval [%l, %u), and the step
    // must be a positive value.
    // For easier conversion, we calculate the trip count and use a canonical
    // induction variable.
    auto diff = mlir::arith::SubIOp::create(rewriter, loc, high, low);
    auto distance = mlir::arith::AddIOp::create(rewriter, loc, diff, step);
    auto tripCount =
        mlir::arith::DivSIOp::create(rewriter, loc, distance, step);
    auto zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto scfForOp =
        mlir::scf::ForOp::create(rewriter, loc, zero, tripCount, one, iterArgs);

    auto &loopOps = doLoopOp.getBody()->getOperations();
    auto resultOp =
        mlir::cast<fir::ResultOp>(doLoopOp.getBody()->getTerminator());
    auto results = resultOp.getOperands();
    mlir::Block *loweredBody = scfForOp.getBody();

    loweredBody->getOperations().splice(loweredBody->begin(), loopOps,
                                        loopOps.begin(),
                                        std::prev(loopOps.end()));

    rewriter.setInsertionPointToStart(loweredBody);
    mlir::Value iv = mlir::arith::MulIOp::create(
        rewriter, loc, scfForOp.getInductionVar(), step);
    iv = mlir::arith::AddIOp::create(rewriter, loc, low, iv);

    if (!results.empty()) {
      rewriter.setInsertionPointToEnd(loweredBody);
      mlir::scf::YieldOp::create(rewriter, resultOp->getLoc(), results);
    }
    doLoopOp.getInductionVar().replaceAllUsesWith(iv);
    rewriter.replaceAllUsesWith(doLoopOp.getRegionIterArgs(),
                                hasFinalValue
                                    ? scfForOp.getRegionIterArgs().drop_front()
                                    : scfForOp.getRegionIterArgs());

    // Copy all the attributes from the old to new op.
    scfForOp->setAttrs(doLoopOp->getAttrs());
    rewriter.replaceOp(doLoopOp, scfForOp);
    return mlir::success();
  }
};

void copyBlockAndTransformResult(mlir::PatternRewriter &rewriter,
                                 mlir::Block &srcBlock, mlir::Block &dstBlock) {
  mlir::Operation *srcTerminator = srcBlock.getTerminator();
  auto resultOp = mlir::cast<fir::ResultOp>(srcTerminator);

  dstBlock.getOperations().splice(dstBlock.begin(), srcBlock.getOperations(),
                                  srcBlock.begin(), std::prev(srcBlock.end()));

  if (!resultOp->getOperands().empty()) {
    rewriter.setInsertionPointToEnd(&dstBlock);
    mlir::scf::YieldOp::create(rewriter, resultOp->getLoc(),
                               resultOp->getOperands());
  }

  rewriter.eraseOp(srcTerminator);
}

struct IfConversion : public mlir::OpRewritePattern<fir::IfOp> {
  using OpRewritePattern<fir::IfOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(fir::IfOp ifOp,
                  mlir::PatternRewriter &rewriter) const override {
    bool hasElse = !ifOp.getElseRegion().empty();
    auto scfIfOp =
        mlir::scf::IfOp::create(rewriter, ifOp.getLoc(), ifOp.getResultTypes(),
                                ifOp.getCondition(), hasElse);

    copyBlockAndTransformResult(rewriter, ifOp.getThenRegion().front(),
                                scfIfOp.getThenRegion().front());

    if (hasElse) {
      copyBlockAndTransformResult(rewriter, ifOp.getElseRegion().front(),
                                  scfIfOp.getElseRegion().front());
    }

    scfIfOp->setAttrs(ifOp->getAttrs());
    rewriter.replaceOp(ifOp, scfIfOp);
    return mlir::success();
  }
};
} // namespace

void FIRToSCFPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<DoLoopConversion, IfConversion>(patterns.getContext());
  mlir::ConversionTarget target(getContext());
  target.addIllegalOp<fir::DoLoopOp, fir::IfOp>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> fir::createFIRToSCFPass() {
  return std::make_unique<FIRToSCFPass>();
}

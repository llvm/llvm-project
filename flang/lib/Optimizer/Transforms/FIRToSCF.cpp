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
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_FIRTOSCFPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class FIRToSCFPass : public fir::impl::FIRToSCFPassBase<FIRToSCFPass> {
  using FIRToSCFPassBase::FIRToSCFPassBase;

public:
  void runOnOperation() override;
};

struct DoLoopConversion : public mlir::OpRewritePattern<fir::DoLoopOp> {
  using OpRewritePattern<fir::DoLoopOp>::OpRewritePattern;

  DoLoopConversion(mlir::MLIRContext *context,
                   bool parallelUnorderedLoop = false,
                   mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<fir::DoLoopOp>(context, benefit),
        parallelUnorderedLoop(parallelUnorderedLoop) {}

  mlir::LogicalResult
  matchAndRewrite(fir::DoLoopOp doLoopOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = doLoopOp.getLoc();
    bool hasFinalValue = doLoopOp.getFinalValue().has_value();
    bool isUnordered = doLoopOp.getUnordered().has_value();

    // Get loop values from the DoLoopOp
    mlir::Value low = doLoopOp.getLowerBound();
    mlir::Value high = doLoopOp.getUpperBound();
    assert(low && high && "must be a Value");
    mlir::Value step = doLoopOp.getStep();
    mlir::SmallVector<mlir::Value> iterArgs;
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

    // Create the scf.for or scf.parallel operation
    mlir::Operation *scfLoopOp = nullptr;
    if (isUnordered && parallelUnorderedLoop) {
      scfLoopOp = mlir::scf::ParallelOp::create(rewriter, loc, {zero},
                                                {tripCount}, {one}, iterArgs);
    } else {
      scfLoopOp = mlir::scf::ForOp::create(rewriter, loc, zero, tripCount, one,
                                           iterArgs);
    }

    // Move the body of the fir.do_loop to the scf.for or scf.parallel
    auto &loopOps = doLoopOp.getBody()->getOperations();
    auto resultOp =
        mlir::cast<fir::ResultOp>(doLoopOp.getBody()->getTerminator());
    auto results = resultOp.getOperands();
    auto scfLoopLikeOp = mlir::cast<mlir::LoopLikeOpInterface>(scfLoopOp);
    mlir::Block &scfLoopBody = scfLoopLikeOp.getLoopRegions().front()->front();

    scfLoopBody.getOperations().splice(scfLoopBody.begin(), loopOps,
                                       loopOps.begin(),
                                       std::prev(loopOps.end()));

    rewriter.setInsertionPointToStart(&scfLoopBody);
    mlir::Value iv = mlir::arith::MulIOp::create(
        rewriter, loc, scfLoopLikeOp.getSingleInductionVar().value(), step);
    iv = mlir::arith::AddIOp::create(rewriter, loc, low, iv);
    mlir::Value firIV = doLoopOp.getInductionVar();
    firIV.replaceAllUsesWith(iv);

    mlir::Value finalValue;
    if (hasFinalValue) {
      // Prefer re-using an existing `arith.addi` in the moved loop body if it
      // already computes the next `iv + step`.
      if (!results.empty()) {
        if (auto addOp = results.front().getDefiningOp<mlir::arith::AddIOp>()) {
          mlir::Value lhs = addOp.getLhs();
          mlir::Value rhs = addOp.getRhs();
          if ((lhs == iv && rhs == step) || (lhs == step && rhs == iv))
            finalValue = results.front();
        }
      }
      if (!finalValue)
        finalValue = mlir::arith::AddIOp::create(rewriter, loc, iv, step);
    }

    if (hasFinalValue || !results.empty()) {
      rewriter.setInsertionPointToEnd(&scfLoopBody);
      llvm::SmallVector<mlir::Value> yieldOperands;
      if (hasFinalValue) {
        yieldOperands.push_back(finalValue);
        llvm::append_range(yieldOperands, results.drop_front());
      } else {
        llvm::append_range(yieldOperands, results);
      }
      mlir::scf::YieldOp::create(rewriter, resultOp->getLoc(), yieldOperands);
    }
    rewriter.replaceAllUsesWith(
        doLoopOp.getRegionIterArgs(),
        hasFinalValue ? scfLoopLikeOp.getRegionIterArgs().drop_front()
                      : scfLoopLikeOp.getRegionIterArgs());

    // Copy loop annotations from the fir.do_loop to scf loop op.
    if (auto ann = doLoopOp.getLoopAnnotation())
      scfLoopOp->setAttr("loop_annotation", *ann);

    rewriter.replaceOp(doLoopOp, scfLoopOp);
    return mlir::success();
  }

private:
  bool parallelUnorderedLoop;
};

struct IterWhileConversion : public mlir::OpRewritePattern<fir::IterWhileOp> {
  using OpRewritePattern<fir::IterWhileOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::IterWhileOp iterWhileOp,
                  mlir::PatternRewriter &rewriter) const override {

    mlir::Location loc = iterWhileOp.getLoc();
    mlir::Value lowerBound = iterWhileOp.getLowerBound();
    mlir::Value upperBound = iterWhileOp.getUpperBound();
    mlir::Value step = iterWhileOp.getStep();

    mlir::Value okInit = iterWhileOp.getIterateIn();
    mlir::ValueRange iterArgs = iterWhileOp.getInitArgs();
    bool hasFinalValue = iterWhileOp.getFinalValue().has_value();

    mlir::SmallVector<mlir::Value> initVals;
    initVals.push_back(lowerBound);
    initVals.push_back(okInit);
    initVals.append(iterArgs.begin(), iterArgs.end());

    mlir::SmallVector<mlir::Type> loopTypes;
    loopTypes.push_back(lowerBound.getType());
    loopTypes.push_back(okInit.getType());
    for (auto val : iterArgs)
      loopTypes.push_back(val.getType());

    auto scfWhileOp =
        mlir::scf::WhileOp::create(rewriter, loc, loopTypes, initVals);

    auto &beforeBlock = *rewriter.createBlock(
        &scfWhileOp.getBefore(), scfWhileOp.getBefore().end(), loopTypes,
        mlir::SmallVector<mlir::Location>(loopTypes.size(), loc));

    mlir::Region::BlockArgListType argsInBefore =
        scfWhileOp.getBefore().getArguments();
    auto ivInBefore = argsInBefore[0];
    auto earlyExitInBefore = argsInBefore[1];

    rewriter.setInsertionPointToStart(&beforeBlock);

    // The comparison depends on the sign of the step value. We fully expect
    // this expression to be folded by the optimizer or LLVM. This expression
    // is written this way so that `step == 0` always returns `false`.
    auto zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto compl0 = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::slt, zero, step);
    auto compl1 = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::sle, ivInBefore, upperBound);
    auto compl2 = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::slt, step, zero);
    auto compl3 = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::sge, ivInBefore, upperBound);
    auto cmp0 = mlir::arith::AndIOp::create(rewriter, loc, compl0, compl1);
    auto cmp1 = mlir::arith::AndIOp::create(rewriter, loc, compl2, compl3);
    auto cmp2 = mlir::arith::OrIOp::create(rewriter, loc, cmp0, cmp1);
    mlir::Value cond =
        mlir::arith::AndIOp::create(rewriter, loc, earlyExitInBefore, cmp2);

    mlir::scf::ConditionOp::create(rewriter, loc, cond, argsInBefore);

    rewriter.moveBlockBefore(iterWhileOp.getBody(), &scfWhileOp.getAfter(),
                             scfWhileOp.getAfter().begin());

    auto *afterBody = scfWhileOp.getAfterBody();
    auto resultOp = mlir::cast<fir::ResultOp>(afterBody->getTerminator());
    mlir::SmallVector<mlir::Value> results;
    mlir::Value iv = scfWhileOp.getAfterArguments()[0];

    rewriter.setInsertionPointToStart(afterBody);
    results.push_back(mlir::arith::AddIOp::create(rewriter, loc, iv, step));
    llvm::append_range(results, hasFinalValue
                                    ? resultOp->getOperands().drop_front()
                                    : resultOp->getOperands());

    rewriter.setInsertionPointToEnd(afterBody);
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(resultOp, results);

    scfWhileOp->setAttrs(iterWhileOp->getAttrs());
    rewriter.replaceOp(iterWhileOp,
                       hasFinalValue ? scfWhileOp->getResults()
                                     : scfWhileOp->getResults().drop_front());
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

void fir::populateFIRToSCFRewrites(mlir::RewritePatternSet &patterns,
                                   bool parallelUnordered) {
  patterns.add<IterWhileConversion, IfConversion>(patterns.getContext());
  patterns.add<DoLoopConversion>(patterns.getContext(), parallelUnordered);
}

void FIRToSCFPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  fir::populateFIRToSCFRewrites(patterns, parallelUnordered);
  walkAndApplyPatterns(getOperation(), std::move(patterns));
}

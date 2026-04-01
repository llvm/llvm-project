//===-- FIRToSCF.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/Transforms/WalkPatternRewriteDriver.h"

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

struct DoLoopConversion : public aiir::OpRewritePattern<fir::DoLoopOp> {
  using OpRewritePattern<fir::DoLoopOp>::OpRewritePattern;

  DoLoopConversion(aiir::AIIRContext *context,
                   bool parallelUnorderedLoop = false,
                   aiir::PatternBenefit benefit = 1)
      : OpRewritePattern<fir::DoLoopOp>(context, benefit),
        parallelUnorderedLoop(parallelUnorderedLoop) {}

  aiir::LogicalResult
  matchAndRewrite(fir::DoLoopOp doLoopOp,
                  aiir::PatternRewriter &rewriter) const override {
    aiir::Location loc = doLoopOp.getLoc();
    bool hasFinalValue = doLoopOp.getFinalValue().has_value();
    bool isUnordered = doLoopOp.getUnordered().has_value();

    // Get loop values from the DoLoopOp
    aiir::Value low = doLoopOp.getLowerBound();
    aiir::Value high = doLoopOp.getUpperBound();
    assert(low && high && "must be a Value");
    aiir::Value step = doLoopOp.getStep();
    aiir::SmallVector<aiir::Value> iterArgs;
    if (hasFinalValue)
      iterArgs.push_back(low);
    iterArgs.append(doLoopOp.getIterOperands().begin(),
                    doLoopOp.getIterOperands().end());

    // fir.do_loop iterates over the interval [%l, %u], and the step may be
    // negative. But scf.for iterates over the interval [%l, %u), and the step
    // must be a positive value.
    // For easier conversion, we calculate the trip count and use a canonical
    // induction variable.
    auto diff = aiir::arith::SubIOp::create(rewriter, loc, high, low);
    auto distance = aiir::arith::AddIOp::create(rewriter, loc, diff, step);
    auto tripCount =
        aiir::arith::DivSIOp::create(rewriter, loc, distance, step);
    auto zero = aiir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto one = aiir::arith::ConstantIndexOp::create(rewriter, loc, 1);

    // Create the scf.for or scf.parallel operation
    aiir::Operation *scfLoopOp = nullptr;
    if (isUnordered && parallelUnorderedLoop) {
      scfLoopOp = aiir::scf::ParallelOp::create(rewriter, loc, {zero},
                                                {tripCount}, {one}, iterArgs);
    } else {
      scfLoopOp = aiir::scf::ForOp::create(rewriter, loc, zero, tripCount, one,
                                           iterArgs);
    }

    // Move the body of the fir.do_loop to the scf.for or scf.parallel
    auto &loopOps = doLoopOp.getBody()->getOperations();
    auto resultOp =
        aiir::cast<fir::ResultOp>(doLoopOp.getBody()->getTerminator());
    auto results = resultOp.getOperands();
    auto scfLoopLikeOp = aiir::cast<aiir::LoopLikeOpInterface>(scfLoopOp);
    aiir::Block &scfLoopBody = scfLoopLikeOp.getLoopRegions().front()->front();

    scfLoopBody.getOperations().splice(scfLoopBody.begin(), loopOps,
                                       loopOps.begin(),
                                       std::prev(loopOps.end()));

    rewriter.setInsertionPointToStart(&scfLoopBody);
    aiir::Value iv = aiir::arith::MulIOp::create(
        rewriter, loc, scfLoopLikeOp.getSingleInductionVar().value(), step);
    iv = aiir::arith::AddIOp::create(rewriter, loc, low, iv);
    aiir::Value firIV = doLoopOp.getInductionVar();
    firIV.replaceAllUsesWith(iv);

    aiir::Value finalValue;
    if (hasFinalValue) {
      // Prefer re-using an existing `arith.addi` in the moved loop body if it
      // already computes the next `iv + step`.
      if (!results.empty()) {
        if (auto addOp = results.front().getDefiningOp<aiir::arith::AddIOp>()) {
          aiir::Value lhs = addOp.getLhs();
          aiir::Value rhs = addOp.getRhs();
          if ((lhs == iv && rhs == step) || (lhs == step && rhs == iv))
            finalValue = results.front();
        }
      }
      if (!finalValue)
        finalValue = aiir::arith::AddIOp::create(rewriter, loc, iv, step);
    }

    if (hasFinalValue || !results.empty()) {
      rewriter.setInsertionPointToEnd(&scfLoopBody);
      llvm::SmallVector<aiir::Value> yieldOperands;
      if (hasFinalValue) {
        yieldOperands.push_back(finalValue);
        llvm::append_range(yieldOperands, results.drop_front());
      } else {
        llvm::append_range(yieldOperands, results);
      }
      aiir::scf::YieldOp::create(rewriter, resultOp->getLoc(), yieldOperands);
    }
    rewriter.replaceAllUsesWith(
        doLoopOp.getRegionIterArgs(),
        hasFinalValue ? scfLoopLikeOp.getRegionIterArgs().drop_front()
                      : scfLoopLikeOp.getRegionIterArgs());

    // Copy loop annotations from the fir.do_loop to scf loop op.
    if (auto ann = doLoopOp.getLoopAnnotation())
      scfLoopOp->setAttr("loop_annotation", *ann);

    rewriter.replaceOp(doLoopOp, scfLoopOp);
    return aiir::success();
  }

private:
  bool parallelUnorderedLoop;
};

struct IterWhileConversion : public aiir::OpRewritePattern<fir::IterWhileOp> {
  using OpRewritePattern<fir::IterWhileOp>::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(fir::IterWhileOp iterWhileOp,
                  aiir::PatternRewriter &rewriter) const override {

    aiir::Location loc = iterWhileOp.getLoc();
    aiir::Value lowerBound = iterWhileOp.getLowerBound();
    aiir::Value upperBound = iterWhileOp.getUpperBound();
    aiir::Value step = iterWhileOp.getStep();

    aiir::Value okInit = iterWhileOp.getIterateIn();
    aiir::ValueRange iterArgs = iterWhileOp.getInitArgs();
    bool hasFinalValue = iterWhileOp.getFinalValue().has_value();

    aiir::SmallVector<aiir::Value> initVals;
    initVals.push_back(lowerBound);
    initVals.push_back(okInit);
    initVals.append(iterArgs.begin(), iterArgs.end());

    aiir::SmallVector<aiir::Type> loopTypes;
    loopTypes.push_back(lowerBound.getType());
    loopTypes.push_back(okInit.getType());
    for (auto val : iterArgs)
      loopTypes.push_back(val.getType());

    auto scfWhileOp =
        aiir::scf::WhileOp::create(rewriter, loc, loopTypes, initVals);

    auto &beforeBlock = *rewriter.createBlock(
        &scfWhileOp.getBefore(), scfWhileOp.getBefore().end(), loopTypes,
        aiir::SmallVector<aiir::Location>(loopTypes.size(), loc));

    aiir::Region::BlockArgListType argsInBefore =
        scfWhileOp.getBefore().getArguments();
    auto ivInBefore = argsInBefore[0];
    auto earlyExitInBefore = argsInBefore[1];

    rewriter.setInsertionPointToStart(&beforeBlock);

    // The comparison depends on the sign of the step value. We fully expect
    // this expression to be folded by the optimizer or LLVM. This expression
    // is written this way so that `step == 0` always returns `false`.
    auto zero = aiir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto compl0 = aiir::arith::CmpIOp::create(
        rewriter, loc, aiir::arith::CmpIPredicate::slt, zero, step);
    auto compl1 = aiir::arith::CmpIOp::create(
        rewriter, loc, aiir::arith::CmpIPredicate::sle, ivInBefore, upperBound);
    auto compl2 = aiir::arith::CmpIOp::create(
        rewriter, loc, aiir::arith::CmpIPredicate::slt, step, zero);
    auto compl3 = aiir::arith::CmpIOp::create(
        rewriter, loc, aiir::arith::CmpIPredicate::sge, ivInBefore, upperBound);
    auto cmp0 = aiir::arith::AndIOp::create(rewriter, loc, compl0, compl1);
    auto cmp1 = aiir::arith::AndIOp::create(rewriter, loc, compl2, compl3);
    auto cmp2 = aiir::arith::OrIOp::create(rewriter, loc, cmp0, cmp1);
    aiir::Value cond =
        aiir::arith::AndIOp::create(rewriter, loc, earlyExitInBefore, cmp2);

    aiir::scf::ConditionOp::create(rewriter, loc, cond, argsInBefore);

    rewriter.moveBlockBefore(iterWhileOp.getBody(), &scfWhileOp.getAfter(),
                             scfWhileOp.getAfter().begin());

    auto *afterBody = scfWhileOp.getAfterBody();
    auto resultOp = aiir::cast<fir::ResultOp>(afterBody->getTerminator());
    aiir::SmallVector<aiir::Value> results;
    aiir::Value iv = scfWhileOp.getAfterArguments()[0];

    rewriter.setInsertionPointToStart(afterBody);
    results.push_back(aiir::arith::AddIOp::create(rewriter, loc, iv, step));
    llvm::append_range(results, hasFinalValue
                                    ? resultOp->getOperands().drop_front()
                                    : resultOp->getOperands());

    rewriter.setInsertionPointToEnd(afterBody);
    rewriter.replaceOpWithNewOp<aiir::scf::YieldOp>(resultOp, results);

    scfWhileOp->setAttrs(iterWhileOp->getAttrs());
    rewriter.replaceOp(iterWhileOp,
                       hasFinalValue ? scfWhileOp->getResults()
                                     : scfWhileOp->getResults().drop_front());
    return aiir::success();
  }
};

void copyBlockAndTransformResult(aiir::PatternRewriter &rewriter,
                                 aiir::Block &srcBlock, aiir::Block &dstBlock) {
  aiir::Operation *srcTerminator = srcBlock.getTerminator();
  auto resultOp = aiir::cast<fir::ResultOp>(srcTerminator);

  dstBlock.getOperations().splice(dstBlock.begin(), srcBlock.getOperations(),
                                  srcBlock.begin(), std::prev(srcBlock.end()));

  if (!resultOp->getOperands().empty()) {
    rewriter.setInsertionPointToEnd(&dstBlock);
    aiir::scf::YieldOp::create(rewriter, resultOp->getLoc(),
                               resultOp->getOperands());
  }

  rewriter.eraseOp(srcTerminator);
}

struct IfConversion : public aiir::OpRewritePattern<fir::IfOp> {
  using OpRewritePattern<fir::IfOp>::OpRewritePattern;
  aiir::LogicalResult
  matchAndRewrite(fir::IfOp ifOp,
                  aiir::PatternRewriter &rewriter) const override {
    bool hasElse = !ifOp.getElseRegion().empty();
    auto scfIfOp =
        aiir::scf::IfOp::create(rewriter, ifOp.getLoc(), ifOp.getResultTypes(),
                                ifOp.getCondition(), hasElse);

    copyBlockAndTransformResult(rewriter, ifOp.getThenRegion().front(),
                                scfIfOp.getThenRegion().front());

    if (hasElse) {
      copyBlockAndTransformResult(rewriter, ifOp.getElseRegion().front(),
                                  scfIfOp.getElseRegion().front());
    }

    scfIfOp->setAttrs(ifOp->getAttrs());
    rewriter.replaceOp(ifOp, scfIfOp);
    return aiir::success();
  }
};
} // namespace

void fir::populateFIRToSCFRewrites(aiir::RewritePatternSet &patterns,
                                   bool parallelUnordered) {
  patterns.add<IterWhileConversion, IfConversion>(patterns.getContext());
  patterns.add<DoLoopConversion>(patterns.getContext(), parallelUnordered);
}

void FIRToSCFPass::runOnOperation() {
  aiir::RewritePatternSet patterns(&getContext());
  fir::populateFIRToSCFRewrites(patterns, parallelUnordered);
  walkAndApplyPatterns(getOperation(), std::move(patterns));
}

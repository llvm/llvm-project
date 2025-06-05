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
    llvm::SmallVector<Value> iterArgs;
    if (hasFinalValue)
      iterArgs.push_back(low);
    iterArgs.append(doLoopOp.getIterOperands().begin(),
                    doLoopOp.getIterOperands().end());

    // fir.do_loop iterates over the interval [%l, %u], and the step may be
    // negative. But scf.for iterates over the interval [%l, %u), and the step
    // must be a positive value.
    // For easier conversion, we calculate the trip count and use a canonical
    // induction variable.
    auto diff = rewriter.create<arith::SubIOp>(loc, high, low);
    auto distance = rewriter.create<arith::AddIOp>(loc, diff, step);
    auto tripCount = rewriter.create<arith::DivSIOp>(loc, distance, step);
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
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

    // Copy all the attributes from the old to new op.
    scfForOp->setAttrs(doLoopOp->getAttrs());
    rewriter.replaceOp(doLoopOp, scfForOp);
    return success();
  }
};

struct IfConversion : public OpRewritePattern<fir::IfOp> {
  using OpRewritePattern<fir::IfOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(fir::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = ifOp.getLoc();
    mlir::detail::TypedValue<mlir::IntegerType> condition = ifOp.getCondition();
    ValueTypeRange<ResultRange> resultTypes = ifOp.getResultTypes();
    bool hasResult = !resultTypes.empty();
    auto scfIfOp = rewriter.create<scf::IfOp>(loc, resultTypes, condition,
                                              !ifOp.getElseRegion().empty());
    // then region
    assert(!ifOp.getThenRegion().empty() && "must have then region");
    auto &firThenBlock = ifOp.getThenRegion().front();
    auto &scfThenBlock = scfIfOp.getThenRegion().front();
    auto &firThenOps = firThenBlock.getOperations();
    mlir::Operation *firThenTerminator = firThenBlock.getTerminator();

    rewriter.setInsertionPointToStart(&scfThenBlock);
    // not splice terminator
    scfThenBlock.getOperations().splice(scfThenBlock.begin(), firThenOps,
                                        firThenOps.begin(),
                                        std::prev(firThenOps.end()));
    // create terminator scf.yield
    if (hasResult) {
      rewriter.setInsertionPointToEnd(&scfThenBlock);
      mlir::OperandRange thenResults = firThenTerminator->getOperands();
      rewriter.create<scf::YieldOp>(firThenTerminator->getLoc(), thenResults);
    }

    // else region
    if (!ifOp.getElseRegion().empty()) {
      auto &firElseBlock = ifOp.getElseRegion().front();
      auto &scfElseBlock = scfIfOp.getElseRegion().front();
      auto &firElseOps = firElseBlock.getOperations();
      mlir::Operation *firElseTerminator = firElseBlock.getTerminator();

      rewriter.setInsertionPointToStart(&scfElseBlock);
      scfElseBlock.getOperations().splice(scfElseBlock.begin(), firElseOps,
                                          firElseOps.begin(),
                                          std::prev(firElseOps.end()));

      if (hasResult) {
        rewriter.setInsertionPointToEnd(&scfElseBlock);
        mlir::OperandRange elseResults = firElseTerminator->getOperands();
        rewriter.create<scf::YieldOp>(firElseTerminator->getLoc(), elseResults);
      }
    }

    scfIfOp->setAttrs(ifOp->getAttrs());
    rewriter.replaceOp(ifOp, scfIfOp);
    return success();
  }
};
} // namespace

void FIRToSCFPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<DoLoopConversion, IfConversion>(patterns.getContext());
  ConversionTarget target(getContext());
  target.addIllegalOp<fir::DoLoopOp, fir::IfOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> fir::createFIRToSCFPass() {
  return std::make_unique<FIRToSCFPass>();
}

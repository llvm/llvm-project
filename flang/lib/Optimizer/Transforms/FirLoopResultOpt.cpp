//===- FirLoopResultOpt.cpp - Optimization pass for fir loops    ------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "flang-fir-result-opt"

namespace {

class LoopResultRemoval : public mlir::OpRewritePattern<fir::DoLoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LoopResultRemoval(mlir::MLIRContext *c) : OpRewritePattern(c) {}
  mlir::LogicalResult
  matchAndRewrite(fir::DoLoopOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    for (auto r : loop.getResults()) {
      if (valueUseful(r))
        return mlir::failure();
    }
    auto &loopOps = loop.getBody()->getOperations();
    auto newLoop = rewriter.create<fir::DoLoopOp>(
        loop.getLoc(), loop.lowerBound(), loop.upperBound(), loop.step());
    rewriter.startRootUpdate(newLoop.getOperation());
    rewriter.startRootUpdate(loop.getOperation());
    newLoop.getBody()->getOperations().splice(
        --newLoop.getBody()->end(), loopOps, loopOps.begin(), --loopOps.end());
    loop.getInductionVar().replaceAllUsesWith(newLoop.getInductionVar());
    rewriter.finalizeRootUpdate(loop.getOperation());
    rewriter.finalizeRootUpdate(newLoop.getOperation());
    for (auto r : loop.getResults()) {
      eraseAllUses(r, rewriter);
    }
    rewriter.eraseBlock(loop.getBody());
    rewriter.eraseOp(loop);
    return mlir::success();
  }

private:
  void eraseAllUses(mlir::Value v, mlir::PatternRewriter &rewriter) const {
    for (auto &use : v.getUses()) {
      if (auto convert = dyn_cast<fir::ConvertOp>(use.getOwner())) {
        eraseAllUses(convert.getResult(), rewriter);
      }
      rewriter.eraseOp(use.getOwner());
    }
  }
  bool valueUseful(mlir::Value v) const {
    for (auto &use : v.getUses()) {
      if (auto convert = dyn_cast<fir::ConvertOp>(use.getOwner()))
        return valueUseful(convert.getResult());
      if (auto store = dyn_cast<fir::StoreOp>(use.getOwner())) {
        bool anyLoad = false;
        for (auto &su : store.memref().getUses()) {
          if (auto load = dyn_cast<fir::LoadOp>(su.getOwner()))
            anyLoad = true;
        }
        return anyLoad;
      }
      return true;
    }
    return false;
  }
};

class FirLoopResultOptPass
    : public fir::FirLoopResultOptBase<FirLoopResultOptPass> {
public:
  void runOnFunction() override {
    auto *context = &getContext();
    auto function = getFunction();
    mlir::OwningRewritePatternList patterns;
    patterns.insert<LoopResultRemoval>(context);
    mlir::ConversionTarget target = *context;
    target.addLegalDialect<fir::FIROpsDialect, mlir::StandardOpsDialect>();
    target.addDynamicallyLegalOp<fir::DoLoopOp>(
        [&](fir::DoLoopOp op) { return op.getNumResults() == 0; });
    if (mlir::failed(mlir::applyPartialConversion(function, target,
                                                  std::move(patterns)))) {
      mlir::emitWarning(mlir::UnknownLoc::get(context),
                        "fir loop result optimization failed\n");
    }
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> fir::createFirLoopResultOptPass() {
  return std::make_unique<FirLoopResultOptPass>();
}

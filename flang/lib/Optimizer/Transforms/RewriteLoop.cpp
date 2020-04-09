//===-- RewriteLoop.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/CommandLine.h"
#include <memory>

/// disable FIR to affine dialect conversion
static llvm::cl::opt<bool>
    disableAffinePromo("disable-affine-promotion",
                       llvm::cl::desc("disable FIR to Affine pass"),
                       llvm::cl::init(false));

/// disable FIR to loop dialect conversion
static llvm::cl::opt<bool>
    disableLoopConversion("disable-loop-conversion",
                          llvm::cl::desc("disable FIR to Loop pass"),
                          llvm::cl::init(false));

namespace fir {
namespace {

template <typename FROM>
class OpRewrite : public mlir::RewritePattern {
public:
  explicit OpRewrite(mlir::MLIRContext *ctx)
      : RewritePattern(FROM::getOperationName(), 1, ctx) {}
};

/// Convert `fir.loop` to `affine.for`
class AffineLoopConv : public OpRewrite<LoopOp> {
public:
  using OpRewrite::OpRewrite;
};

/// Convert `fir.where` to `affine.if`
class AffineWhereConv : public OpRewrite<WhereOp> {
public:
  using OpRewrite::OpRewrite;
};

/// Promote fir.loop and fir.where to affine.for and affine.if, in the cases
/// where such a promotion is possible.
class AffineDialectPromotion
    : public mlir::PassWrapper<AffineDialectPromotion, mlir::FunctionPass> {
public:
  void runOnFunction() override {
    if (disableAffinePromo)
      return;

    auto *context = &getContext();
    mlir::OwningRewritePatternList patterns;
    patterns.insert<AffineLoopConv, AffineWhereConv>(context);
    mlir::ConversionTarget target = *context;
    target.addLegalDialect<mlir::AffineDialect, FIROpsDialect,
                           mlir::loop::LoopOpsDialect,
                           mlir::StandardOpsDialect>();
    // target.addDynamicallyLegalOp<LoopOp, WhereOp>();

    // apply the patterns
    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting to affine dialect\n");
      signalPassFailure();
    }
  }
};

// Conversion to the MLIR loop dialect
//
// FIR loops that cannot be converted to the affine dialect will remain as
// `fir.loop` operations.  These can be converted to `loop.for` operations. MLIR
// includes a pass to lower `loop.for` operations to a CFG.

/// Convert `fir.loop` to `loop.for`
class LoopLoopConv : public mlir::OpRewritePattern<LoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(LoopOp loop, mlir::PatternRewriter &rewriter) const override {
    auto loc = loop.getLoc();
    auto low = loop.getLowerBoundOperand();
    if (!low) {
      assert(loop.constantLowerBound().hasValue());
      auto lb = *loop.constantLowerBound();
      low = rewriter.create<mlir::ConstantIndexOp>(loc, lb.getSExtValue());
    }
    auto high = loop.getUpperBoundOperand();
    if (!high) {
      assert(loop.constantUpperBound().hasValue());
      auto ub = *loop.constantUpperBound();
      high = rewriter.create<mlir::ConstantIndexOp>(loc, ub.getSExtValue());
    }
    auto step = loop.getStepOperand();
    if (!step) {
      if (loop.constantStep().hasValue()) {
        auto st = *loop.constantStep();
        step = rewriter.create<mlir::ConstantIndexOp>(loc, st.getSExtValue());
      } else {
        step = rewriter.create<mlir::ConstantIndexOp>(loc, 1);
      }
    }
    assert(low && high && step);
    // ForOp has different bounds semantics. Adjust upper bound.
    auto adjustUp = rewriter.create<mlir::AddIOp>(loc, high, step);
    auto f = rewriter.create<mlir::loop::ForOp>(loc, low, adjustUp, step);
    f.region().getBlocks().clear();
    rewriter.inlineRegionBefore(loop.region(), f.region(), f.region().end());
    rewriter.eraseOp(loop);
    return success();
  }
};

/// Convert `fir.where` to `loop.if`
class LoopWhereConv : public mlir::OpRewritePattern<WhereOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(WhereOp where,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = where.getLoc();
    bool hasOtherRegion = !where.otherRegion().empty();
    auto cond = where.condition();
    auto ifOp = rewriter.create<mlir::loop::IfOp>(loc, cond, hasOtherRegion);
    rewriter.inlineRegionBefore(where.whereRegion(), &ifOp.thenRegion().back());
    ifOp.thenRegion().back().erase();
    if (hasOtherRegion) {
      rewriter.inlineRegionBefore(where.otherRegion(),
                                  &ifOp.elseRegion().back());
      ifOp.elseRegion().back().erase();
    }
    rewriter.eraseOp(where);
    return success();
  }
};

/// Replace FirEndOp with TerminatorOp
class LoopFirEndConv : public mlir::OpRewritePattern<FirEndOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(FirEndOp op, mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::loop::YieldOp>(op);
    return success();
  }
};

/// Convert `fir.loop` and `fir.where` to `loop.for` and `loop.if`.
class LoopDialectConversion
    : public mlir::PassWrapper<LoopDialectConversion, mlir::FunctionPass> {
public:
  void runOnFunction() override {
    if (disableLoopConversion)
      return;

    auto *context = &getContext();
    mlir::OwningRewritePatternList patterns;
    patterns.insert<LoopLoopConv, LoopWhereConv, LoopFirEndConv>(context);
    mlir::ConversionTarget target = *context;
    target.addLegalDialect<mlir::AffineDialect, FIROpsDialect,
                           mlir::loop::LoopOpsDialect,
                           mlir::StandardOpsDialect>();
    target.addIllegalOp<FirEndOp, LoopOp, WhereOp>();

    // apply the patterns
    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting to MLIR loop dialect\n");
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace fir

/// Convert FIR loop constructs to the Affine dialect
std::unique_ptr<mlir::Pass> fir::createPromoteToAffinePass() {
  return std::make_unique<AffineDialectPromotion>();
}

/// Convert `fir.loop` and `fir.where` to `loop.for` and `loop.if`.  This
/// conversion enables the `createLowerToCFGPass` to transform these to CFG
/// form.
std::unique_ptr<mlir::Pass> fir::createLowerToLoopPass() {
  return std::make_unique<LoopDialectConversion>();
}

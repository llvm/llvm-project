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
    auto low = loop.lowerBound();
    auto high = loop.upperBound();
    auto step = loop.step();
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
class LoopResultConv : public mlir::OpRewritePattern<ResultOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::ResultOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::loop::YieldOp>(op);
    return success();
  }
};

class LoopIterWhileConv : public mlir::OpRewritePattern<fir::IterWhileOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::IterWhileOp whileOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = whileOp.getLoc();

    // Start by splitting the block containing the 'fir.do_loop' into two parts.
    // The part before will get the init code, the part after will be the end
    // point.
    auto *initBlock = rewriter.getInsertionBlock();
    auto initPosition = rewriter.getInsertionPoint();
    auto *endBlock = rewriter.splitBlock(initBlock, initPosition);

    // Use the first block of the loop body as the condition block since it is
    // the block that has the induction variable and loop-carried values as
    // arguments. Split out all operations from the first block into a new
    // block. Move all body blocks from the loop body region to the region
    // containing the loop.
    auto *conditionBlock = &whileOp.region().front();
    auto *firstBodyBlock =
        rewriter.splitBlock(conditionBlock, conditionBlock->begin());
    auto *lastBodyBlock = &whileOp.region().back();
    rewriter.inlineRegionBefore(whileOp.region(), endBlock);
    auto iv = conditionBlock->getArgument(0);
    auto iterateVar = conditionBlock->getArgument(1);

    // Append the induction variable stepping logic to the last body block and
    // branch back to the condition block. Loop-carried values are taken from
    // operands of the loop terminator.
    mlir::Operation *terminator = lastBodyBlock->getTerminator();
    rewriter.setInsertionPointToEnd(lastBodyBlock);
    auto step = whileOp.step();
    auto stepped = rewriter.create<AddIOp>(loc, iv, step).getResult();
    if (!stepped)
      return failure();

    llvm::SmallVector<mlir::Value, 8> loopCarried;
    loopCarried.push_back(stepped);
    loopCarried.append(terminator->operand_begin(), terminator->operand_end());
    rewriter.create<BranchOp>(loc, conditionBlock, loopCarried);
    rewriter.eraseOp(terminator);

    // Compute loop bounds before branching to the condition.
    rewriter.setInsertionPointToEnd(initBlock);
    mlir::Value lowerBound = whileOp.lowerBound();
    mlir::Value upperBound = whileOp.upperBound();
    if (!lowerBound || !upperBound)
      return failure();

    // The initial values of loop-carried values is obtained from the operands
    // of the loop operation.
    llvm::SmallVector<mlir::Value, 8> destOperands;
    destOperands.push_back(lowerBound);
    auto iterOperands = whileOp.getIterOperands();
    destOperands.append(iterOperands.begin(), iterOperands.end());
    rewriter.create<BranchOp>(loc, conditionBlock, destOperands);

    // With the body block done, we can fill in the condition block.
    rewriter.setInsertionPointToEnd(conditionBlock);
    auto comp1 =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, iv, upperBound);
    // Remember to AND in the early-exit bool.
    auto comparison = rewriter.create<AndOp>(loc, comp1, iterateVar);
    rewriter.create<CondBranchOp>(loc, comparison, firstBodyBlock,
                                  ArrayRef<Value>(), endBlock,
                                  ArrayRef<Value>());
    // The result of the loop operation is the values of the condition block
    // arguments except the induction variable on the last iteration.
    rewriter.replaceOp(whileOp, conditionBlock->getArguments().drop_front());
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
    mlir::OwningRewritePatternList patterns1;
    patterns1.insert<LoopIterWhileConv>(context);

    mlir::OwningRewritePatternList patterns2;
    patterns2.insert<LoopLoopConv, LoopWhereConv, LoopResultConv>(context);
    mlir::ConversionTarget target = *context;
    target.addLegalDialect<mlir::AffineDialect, FIROpsDialect,
                           mlir::loop::LoopOpsDialect,
                           mlir::StandardOpsDialect>();

    // apply the patterns
    target.addIllegalOp<IterWhileOp>();
    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target,
                                                  std::move(patterns1)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting to CFG\n");
      signalPassFailure();
    }
    target.addIllegalOp<ResultOp, LoopOp, WhereOp>();
    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target,
                                                  std::move(patterns2)))) {
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

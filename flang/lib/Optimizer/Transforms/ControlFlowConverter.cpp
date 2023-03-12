//===-- ControlFlowConverter.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/TypeCode.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Runtime/derived-api.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"

namespace fir {
#define GEN_PASS_DEF_CFGCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace mlir;

namespace {

// Conversion of fir control ops to more primitive control-flow.
//
// FIR loops that cannot be converted to the affine dialect will remain as
// `fir.do_loop` operations.  These can be converted to control-flow operations.

/// Convert `fir.do_loop` to CFG
class CfgLoopConv : public mlir::OpRewritePattern<fir::DoLoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  CfgLoopConv(mlir::MLIRContext *ctx, bool forceLoopToExecuteOnce)
      : mlir::OpRewritePattern<fir::DoLoopOp>(ctx),
        forceLoopToExecuteOnce(forceLoopToExecuteOnce) {}

  mlir::LogicalResult
  matchAndRewrite(DoLoopOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = loop.getLoc();

    // Create the start and end blocks that will wrap the DoLoopOp with an
    // initalizer and an end point
    auto *initBlock = rewriter.getInsertionBlock();
    auto initPos = rewriter.getInsertionPoint();
    auto *endBlock = rewriter.splitBlock(initBlock, initPos);

    // Split the first DoLoopOp block in two parts. The part before will be the
    // conditional block since it already has the induction variable and
    // loop-carried values as arguments.
    auto *conditionalBlock = &loop.getRegion().front();
    conditionalBlock->addArgument(rewriter.getIndexType(), loc);
    auto *firstBlock =
        rewriter.splitBlock(conditionalBlock, conditionalBlock->begin());
    auto *lastBlock = &loop.getRegion().back();

    // Move the blocks from the DoLoopOp between initBlock and endBlock
    rewriter.inlineRegionBefore(loop.getRegion(), endBlock);

    // Get loop values from the DoLoopOp
    auto low = loop.getLowerBound();
    auto high = loop.getUpperBound();
    assert(low && high && "must be a Value");
    auto step = loop.getStep();

    // Initalization block
    rewriter.setInsertionPointToEnd(initBlock);
    auto diff = rewriter.create<mlir::arith::SubIOp>(loc, high, low);
    auto distance = rewriter.create<mlir::arith::AddIOp>(loc, diff, step);
    mlir::Value iters =
        rewriter.create<mlir::arith::DivSIOp>(loc, distance, step);

    if (forceLoopToExecuteOnce) {
      auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto cond = rewriter.create<mlir::arith::CmpIOp>(
          loc, arith::CmpIPredicate::sle, iters, zero);
      auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      iters = rewriter.create<mlir::arith::SelectOp>(loc, cond, one, iters);
    }

    llvm::SmallVector<mlir::Value> loopOperands;
    loopOperands.push_back(low);
    auto operands = loop.getIterOperands();
    loopOperands.append(operands.begin(), operands.end());
    loopOperands.push_back(iters);

    rewriter.create<mlir::cf::BranchOp>(loc, conditionalBlock, loopOperands);

    // Last loop block
    auto *terminator = lastBlock->getTerminator();
    rewriter.setInsertionPointToEnd(lastBlock);
    auto iv = conditionalBlock->getArgument(0);
    mlir::Value steppedIndex =
        rewriter.create<mlir::arith::AddIOp>(loc, iv, step);
    assert(steppedIndex && "must be a Value");
    auto lastArg = conditionalBlock->getNumArguments() - 1;
    auto itersLeft = conditionalBlock->getArgument(lastArg);
    auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value itersMinusOne =
        rewriter.create<mlir::arith::SubIOp>(loc, itersLeft, one);

    llvm::SmallVector<mlir::Value> loopCarried;
    loopCarried.push_back(steppedIndex);
    auto begin = loop.getFinalValue() ? std::next(terminator->operand_begin())
                                      : terminator->operand_begin();
    loopCarried.append(begin, terminator->operand_end());
    loopCarried.push_back(itersMinusOne);
    rewriter.create<mlir::cf::BranchOp>(loc, conditionalBlock, loopCarried);
    rewriter.eraseOp(terminator);

    // Conditional block
    rewriter.setInsertionPointToEnd(conditionalBlock);
    auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto comparison = rewriter.create<mlir::arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, itersLeft, zero);

    rewriter.create<mlir::cf::CondBranchOp>(
        loc, comparison, firstBlock, llvm::ArrayRef<mlir::Value>(), endBlock,
        llvm::ArrayRef<mlir::Value>());

    // The result of the loop operation is the values of the condition block
    // arguments except the induction variable on the last iteration.
    auto args = loop.getFinalValue()
                    ? conditionalBlock->getArguments()
                    : conditionalBlock->getArguments().drop_front();
    rewriter.replaceOp(loop, args.drop_back());
    return success();
  }

private:
  bool forceLoopToExecuteOnce;
};

/// Convert `fir.if` to control-flow
class CfgIfConv : public mlir::OpRewritePattern<fir::IfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  CfgIfConv(mlir::MLIRContext *ctx, bool forceLoopToExecuteOnce)
      : mlir::OpRewritePattern<fir::IfOp>(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(IfOp ifOp, mlir::PatternRewriter &rewriter) const override {
    auto loc = ifOp.getLoc();

    // Split the block containing the 'fir.if' into two parts.  The part before
    // will contain the condition, the part after will be the continuation
    // point.
    auto *condBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
    mlir::Block *continueBlock;
    if (ifOp.getNumResults() == 0) {
      continueBlock = remainingOpsBlock;
    } else {
      continueBlock = rewriter.createBlock(
          remainingOpsBlock, ifOp.getResultTypes(),
          llvm::SmallVector<mlir::Location>(ifOp.getNumResults(), loc));
      rewriter.create<mlir::cf::BranchOp>(loc, remainingOpsBlock);
    }

    // Move blocks from the "then" region to the region containing 'fir.if',
    // place it before the continuation block, and branch to it.
    auto &ifOpRegion = ifOp.getThenRegion();
    auto *ifOpBlock = &ifOpRegion.front();
    auto *ifOpTerminator = ifOpRegion.back().getTerminator();
    auto ifOpTerminatorOperands = ifOpTerminator->getOperands();
    rewriter.setInsertionPointToEnd(&ifOpRegion.back());
    rewriter.create<mlir::cf::BranchOp>(loc, continueBlock,
                                        ifOpTerminatorOperands);
    rewriter.eraseOp(ifOpTerminator);
    rewriter.inlineRegionBefore(ifOpRegion, continueBlock);

    // Move blocks from the "else" region (if present) to the region containing
    // 'fir.if', place it before the continuation block and branch to it.  It
    // will be placed after the "then" regions.
    auto *otherwiseBlock = continueBlock;
    auto &otherwiseRegion = ifOp.getElseRegion();
    if (!otherwiseRegion.empty()) {
      otherwiseBlock = &otherwiseRegion.front();
      auto *otherwiseTerm = otherwiseRegion.back().getTerminator();
      auto otherwiseTermOperands = otherwiseTerm->getOperands();
      rewriter.setInsertionPointToEnd(&otherwiseRegion.back());
      rewriter.create<mlir::cf::BranchOp>(loc, continueBlock,
                                          otherwiseTermOperands);
      rewriter.eraseOp(otherwiseTerm);
      rewriter.inlineRegionBefore(otherwiseRegion, continueBlock);
    }

    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<mlir::cf::CondBranchOp>(
        loc, ifOp.getCondition(), ifOpBlock, llvm::ArrayRef<mlir::Value>(),
        otherwiseBlock, llvm::ArrayRef<mlir::Value>());
    rewriter.replaceOp(ifOp, continueBlock->getArguments());
    return success();
  }
};

/// Convert `fir.iter_while` to control-flow.
class CfgIterWhileConv : public mlir::OpRewritePattern<fir::IterWhileOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  CfgIterWhileConv(mlir::MLIRContext *ctx, bool forceLoopToExecuteOnce)
      : mlir::OpRewritePattern<fir::IterWhileOp>(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(fir::IterWhileOp whileOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = whileOp.getLoc();

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
    auto *conditionBlock = &whileOp.getRegion().front();
    auto *firstBodyBlock =
        rewriter.splitBlock(conditionBlock, conditionBlock->begin());
    auto *lastBodyBlock = &whileOp.getRegion().back();
    rewriter.inlineRegionBefore(whileOp.getRegion(), endBlock);
    auto iv = conditionBlock->getArgument(0);
    auto iterateVar = conditionBlock->getArgument(1);

    // Append the induction variable stepping logic to the last body block and
    // branch back to the condition block. Loop-carried values are taken from
    // operands of the loop terminator.
    auto *terminator = lastBodyBlock->getTerminator();
    rewriter.setInsertionPointToEnd(lastBodyBlock);
    auto step = whileOp.getStep();
    mlir::Value stepped = rewriter.create<mlir::arith::AddIOp>(loc, iv, step);
    assert(stepped && "must be a Value");

    llvm::SmallVector<mlir::Value> loopCarried;
    loopCarried.push_back(stepped);
    auto begin = whileOp.getFinalValue()
                     ? std::next(terminator->operand_begin())
                     : terminator->operand_begin();
    loopCarried.append(begin, terminator->operand_end());
    rewriter.create<mlir::cf::BranchOp>(loc, conditionBlock, loopCarried);
    rewriter.eraseOp(terminator);

    // Compute loop bounds before branching to the condition.
    rewriter.setInsertionPointToEnd(initBlock);
    auto lowerBound = whileOp.getLowerBound();
    auto upperBound = whileOp.getUpperBound();
    assert(lowerBound && upperBound && "must be a Value");

    // The initial values of loop-carried values is obtained from the operands
    // of the loop operation.
    llvm::SmallVector<mlir::Value> destOperands;
    destOperands.push_back(lowerBound);
    auto iterOperands = whileOp.getIterOperands();
    destOperands.append(iterOperands.begin(), iterOperands.end());
    rewriter.create<mlir::cf::BranchOp>(loc, conditionBlock, destOperands);

    // With the body block done, we can fill in the condition block.
    rewriter.setInsertionPointToEnd(conditionBlock);
    // The comparison depends on the sign of the step value. We fully expect
    // this expression to be folded by the optimizer or LLVM. This expression
    // is written this way so that `step == 0` always returns `false`.
    auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto compl0 = rewriter.create<mlir::arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, zero, step);
    auto compl1 = rewriter.create<mlir::arith::CmpIOp>(
        loc, arith::CmpIPredicate::sle, iv, upperBound);
    auto compl2 = rewriter.create<mlir::arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, step, zero);
    auto compl3 = rewriter.create<mlir::arith::CmpIOp>(
        loc, arith::CmpIPredicate::sle, upperBound, iv);
    auto cmp0 = rewriter.create<mlir::arith::AndIOp>(loc, compl0, compl1);
    auto cmp1 = rewriter.create<mlir::arith::AndIOp>(loc, compl2, compl3);
    auto cmp2 = rewriter.create<mlir::arith::OrIOp>(loc, cmp0, cmp1);
    // Remember to AND in the early-exit bool.
    auto comparison =
        rewriter.create<mlir::arith::AndIOp>(loc, iterateVar, cmp2);
    rewriter.create<mlir::cf::CondBranchOp>(
        loc, comparison, firstBodyBlock, llvm::ArrayRef<mlir::Value>(),
        endBlock, llvm::ArrayRef<mlir::Value>());
    // The result of the loop operation is the values of the condition block
    // arguments except the induction variable on the last iteration.
    auto args = whileOp.getFinalValue()
                    ? conditionBlock->getArguments()
                    : conditionBlock->getArguments().drop_front();
    rewriter.replaceOp(whileOp, args);
    return success();
  }
};

/// Convert FIR structured control flow ops to CFG ops.
class CfgConversion : public fir::impl::CFGConversionBase<CfgConversion> {
public:
  void runOnOperation() override {
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<CfgLoopConv, CfgIfConv, CfgIterWhileConv>(
        context, forceLoopToExecuteOnce);
    mlir::ConversionTarget target(*context);
    target.addLegalDialect<mlir::AffineDialect, mlir::cf::ControlFlowDialect,
                           FIROpsDialect, mlir::func::FuncDialect>();

    // apply the patterns
    target.addIllegalOp<ResultOp, DoLoopOp, IfOp, IterWhileOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting to CFG\n");
      signalPassFailure();
    }
  }
};
} // namespace

/// Convert FIR's structured control flow ops to CFG ops.  This
/// conversion enables the `createLowerToCFGPass` to transform these to CFG
/// form.
std::unique_ptr<mlir::Pass> fir::createFirToCfgPass() {
  return std::make_unique<CfgConversion>();
}

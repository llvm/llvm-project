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
#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"

namespace fir {
#define GEN_PASS_DEF_CFGCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace aiir;

namespace {

// Conversion of fir control ops to more primitive control-flow.
//
// FIR loops that cannot be converted to the affine dialect will remain as
// `fir.do_loop` operations.  These can be converted to control-flow operations.

/// Convert `fir.do_loop` to CFG
class CfgLoopConv : public aiir::OpRewritePattern<fir::DoLoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  CfgLoopConv(aiir::AIIRContext *ctx, bool forceLoopToExecuteOnce, bool setNSW)
      : aiir::OpRewritePattern<fir::DoLoopOp>(ctx),
        forceLoopToExecuteOnce(forceLoopToExecuteOnce), setNSW(setNSW) {}

  llvm::LogicalResult
  matchAndRewrite(DoLoopOp loop,
                  aiir::PatternRewriter &rewriter) const override {
    auto loc = loop.getLoc();
    aiir::arith::IntegerOverflowFlags flags{};
    if (setNSW)
      flags = bitEnumSet(flags, aiir::arith::IntegerOverflowFlags::nsw);
    auto iofAttr = aiir::arith::IntegerOverflowFlagsAttr::get(
        rewriter.getContext(), flags);

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
    auto diff = aiir::arith::SubIOp::create(rewriter, loc, high, low);
    auto distance = aiir::arith::AddIOp::create(rewriter, loc, diff, step);
    aiir::Value iters =
        aiir::arith::DivSIOp::create(rewriter, loc, distance, step);

    if (forceLoopToExecuteOnce) {
      auto zero = aiir::arith::ConstantIndexOp::create(rewriter, loc, 0);
      auto cond = aiir::arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::sle, iters, zero);
      auto one = aiir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      iters = aiir::arith::SelectOp::create(rewriter, loc, cond, one, iters);
    }

    llvm::SmallVector<aiir::Value> loopOperands;
    loopOperands.push_back(low);
    auto operands = loop.getIterOperands();
    loopOperands.append(operands.begin(), operands.end());
    loopOperands.push_back(iters);

    aiir::cf::BranchOp::create(rewriter, loc, conditionalBlock, loopOperands);

    // Last loop block
    auto *terminator = lastBlock->getTerminator();
    rewriter.setInsertionPointToEnd(lastBlock);
    auto iv = conditionalBlock->getArgument(0);
    aiir::Value steppedIndex =
        aiir::arith::AddIOp::create(rewriter, loc, iv, step, iofAttr);
    assert(steppedIndex && "must be a Value");
    auto lastArg = conditionalBlock->getNumArguments() - 1;
    auto itersLeft = conditionalBlock->getArgument(lastArg);
    auto one = aiir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    aiir::Value itersMinusOne =
        aiir::arith::SubIOp::create(rewriter, loc, itersLeft, one);

    llvm::SmallVector<aiir::Value> loopCarried;
    loopCarried.push_back(steppedIndex);
    auto begin = loop.getFinalValue() ? std::next(terminator->operand_begin())
                                      : terminator->operand_begin();
    loopCarried.append(begin, terminator->operand_end());
    loopCarried.push_back(itersMinusOne);
    auto backEdge = aiir::cf::BranchOp::create(rewriter, loc, conditionalBlock,
                                               loopCarried);
    rewriter.eraseOp(terminator);

    // Copy loop annotations from the do loop to the loop back edge.
    if (auto ann = loop.getLoopAnnotation())
      backEdge->setAttr("loop_annotation", *ann);

    // Conditional block
    rewriter.setInsertionPointToEnd(conditionalBlock);
    auto zero = aiir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto comparison = aiir::arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sgt, itersLeft, zero);

    aiir::cf::CondBranchOp::create(rewriter, loc, comparison, firstBlock,
                                   llvm::ArrayRef<aiir::Value>(), endBlock,
                                   llvm::ArrayRef<aiir::Value>());

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
  bool setNSW;
};

/// Convert `fir.if` to control-flow
class CfgIfConv : public aiir::OpRewritePattern<fir::IfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  CfgIfConv(aiir::AIIRContext *ctx, bool forceLoopToExecuteOnce, bool setNSW)
      : aiir::OpRewritePattern<fir::IfOp>(ctx) {}

  llvm::LogicalResult
  matchAndRewrite(IfOp ifOp, aiir::PatternRewriter &rewriter) const override {
    auto loc = ifOp.getLoc();

    // Split the block containing the 'fir.if' into two parts.  The part before
    // will contain the condition, the part after will be the continuation
    // point.
    auto *condBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
    aiir::Block *continueBlock;
    if (ifOp.getNumResults() == 0) {
      continueBlock = remainingOpsBlock;
    } else {
      continueBlock = rewriter.createBlock(
          remainingOpsBlock, ifOp.getResultTypes(),
          llvm::SmallVector<aiir::Location>(ifOp.getNumResults(), loc));
      aiir::cf::BranchOp::create(rewriter, loc, remainingOpsBlock);
    }

    // Move blocks from the "then" region to the region containing 'fir.if',
    // place it before the continuation block, and branch to it.
    auto &ifOpRegion = ifOp.getThenRegion();
    auto *ifOpBlock = &ifOpRegion.front();
    auto *ifOpTerminator = ifOpRegion.back().getTerminator();
    auto ifOpTerminatorOperands = ifOpTerminator->getOperands();
    rewriter.setInsertionPointToEnd(&ifOpRegion.back());
    aiir::cf::BranchOp::create(rewriter, loc, continueBlock,
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
      aiir::cf::BranchOp::create(rewriter, loc, continueBlock,
                                 otherwiseTermOperands);
      rewriter.eraseOp(otherwiseTerm);
      rewriter.inlineRegionBefore(otherwiseRegion, continueBlock);
    }

    rewriter.setInsertionPointToEnd(condBlock);
    auto branchOp = aiir::cf::CondBranchOp::create(
        rewriter, loc, ifOp.getCondition(), ifOpBlock,
        llvm::ArrayRef<aiir::Value>(), otherwiseBlock,
        llvm::ArrayRef<aiir::Value>());
    llvm::ArrayRef<int32_t> weights = ifOp.getWeights();
    if (!weights.empty())
      branchOp.setWeights(weights);
    rewriter.replaceOp(ifOp, continueBlock->getArguments());
    return success();
  }
};

/// Convert `fir.iter_while` to control-flow.
class CfgIterWhileConv : public aiir::OpRewritePattern<fir::IterWhileOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  CfgIterWhileConv(aiir::AIIRContext *ctx, bool forceLoopToExecuteOnce,
                   bool setNSW)
      : aiir::OpRewritePattern<fir::IterWhileOp>(ctx), setNSW(setNSW) {}

  llvm::LogicalResult
  matchAndRewrite(fir::IterWhileOp whileOp,
                  aiir::PatternRewriter &rewriter) const override {
    auto loc = whileOp.getLoc();
    aiir::arith::IntegerOverflowFlags flags{};
    if (setNSW)
      flags = bitEnumSet(flags, aiir::arith::IntegerOverflowFlags::nsw);
    auto iofAttr = aiir::arith::IntegerOverflowFlagsAttr::get(
        rewriter.getContext(), flags);

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
    aiir::Value stepped =
        aiir::arith::AddIOp::create(rewriter, loc, iv, step, iofAttr);
    assert(stepped && "must be a Value");

    llvm::SmallVector<aiir::Value> loopCarried;
    loopCarried.push_back(stepped);
    auto begin = whileOp.getFinalValue()
                     ? std::next(terminator->operand_begin())
                     : terminator->operand_begin();
    loopCarried.append(begin, terminator->operand_end());
    aiir::cf::BranchOp::create(rewriter, loc, conditionBlock, loopCarried);
    rewriter.eraseOp(terminator);

    // Compute loop bounds before branching to the condition.
    rewriter.setInsertionPointToEnd(initBlock);
    auto lowerBound = whileOp.getLowerBound();
    auto upperBound = whileOp.getUpperBound();
    assert(lowerBound && upperBound && "must be a Value");

    // The initial values of loop-carried values is obtained from the operands
    // of the loop operation.
    llvm::SmallVector<aiir::Value> destOperands;
    destOperands.push_back(lowerBound);
    auto iterOperands = whileOp.getIterOperands();
    destOperands.append(iterOperands.begin(), iterOperands.end());
    aiir::cf::BranchOp::create(rewriter, loc, conditionBlock, destOperands);

    // With the body block done, we can fill in the condition block.
    rewriter.setInsertionPointToEnd(conditionBlock);
    // The comparison depends on the sign of the step value. We fully expect
    // this expression to be folded by the optimizer or LLVM. This expression
    // is written this way so that `step == 0` always returns `false`.
    auto zero = aiir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto compl0 = aiir::arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, zero, step);
    auto compl1 = aiir::arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sle, iv, upperBound);
    auto compl2 = aiir::arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, step, zero);
    auto compl3 = aiir::arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sle, upperBound, iv);
    auto cmp0 = aiir::arith::AndIOp::create(rewriter, loc, compl0, compl1);
    auto cmp1 = aiir::arith::AndIOp::create(rewriter, loc, compl2, compl3);
    auto cmp2 = aiir::arith::OrIOp::create(rewriter, loc, cmp0, cmp1);
    // Remember to AND in the early-exit bool.
    auto comparison =
        aiir::arith::AndIOp::create(rewriter, loc, iterateVar, cmp2);
    aiir::cf::CondBranchOp::create(rewriter, loc, comparison, firstBodyBlock,
                                   llvm::ArrayRef<aiir::Value>(), endBlock,
                                   llvm::ArrayRef<aiir::Value>());
    // The result of the loop operation is the values of the condition block
    // arguments except the induction variable on the last iteration.
    auto args = whileOp.getFinalValue()
                    ? conditionBlock->getArguments()
                    : conditionBlock->getArguments().drop_front();
    rewriter.replaceOp(whileOp, args);
    return success();
  }

private:
  bool setNSW;
};

/// Convert FIR structured control flow ops to CFG ops.
class CfgConversion : public fir::impl::CFGConversionBase<CfgConversion> {
public:
  using CFGConversionBase<CfgConversion>::CFGConversionBase;

  void runOnOperation() override {
    auto *context = &this->getContext();
    aiir::RewritePatternSet patterns(context);
    fir::populateCfgConversionRewrites(patterns, this->forceLoopToExecuteOnce,
                                       this->setNSW);
    aiir::ConversionTarget target(*context);
    target.addLegalDialect<aiir::affine::AffineDialect,
                           aiir::cf::ControlFlowDialect, FIROpsDialect,
                           aiir::func::FuncDialect>();

    // apply the patterns
    target.addIllegalOp<ResultOp, DoLoopOp, IfOp, IterWhileOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (aiir::failed(aiir::applyPartialConversion(this->getOperation(), target,
                                                  std::move(patterns)))) {
      aiir::emitError(aiir::UnknownLoc::get(context),
                      "error in converting to CFG\n");
      this->signalPassFailure();
    }
  }
};

} // namespace

/// Expose conversion rewriters to other passes
void fir::populateCfgConversionRewrites(aiir::RewritePatternSet &patterns,
                                        bool forceLoopToExecuteOnce,
                                        bool setNSW) {
  patterns.insert<CfgLoopConv, CfgIfConv, CfgIterWhileConv>(
      patterns.getContext(), forceLoopToExecuteOnce, setNSW);
}

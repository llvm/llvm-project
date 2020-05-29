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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/CommandLine.h"

/// disable FIR to scf dialect conversion
static llvm::cl::opt<bool>
    disableScfConversion("disable-scf-conversion",
                         llvm::cl::desc("disable FIR to SCF pass"),
                         llvm::cl::init(false));

using namespace fir;

namespace {

// Conversion to CFG
class ScfLoopConv : public mlir::OpRewritePattern<fir::LoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // FIXME: This should NOT be using scf.for. Instead, fir.do_loop should be
  // lowered to a semantically correct CFG.
  mlir::LogicalResult
  matchAndRewrite(LoopOp loop, mlir::PatternRewriter &rewriter) const override {
    Location loc = loop.getLoc();

    // Create the start and end blocks that will wrap the LoopOp with an
    // initalizer and an end point
    mlir::Block *initBlock = rewriter.getInsertionBlock();
    auto initPos = rewriter.getInsertionPoint();
    mlir::Block *endBlock = rewriter.splitBlock(initBlock, initPos);

    // Split the first LoopOp block in two parts. The part before will be the
    // conditional block since it already has the induction variable and
    // loop-carried values as arguments.
    mlir::Block *conditionalBlock = &loop.region().front();
    mlir::Block *firstBlock = rewriter.splitBlock(conditionalBlock,
                                                  conditionalBlock->begin());
    mlir::Block *lastBlock = &loop.region().back();

    // Move the blocks from the LoopOp between initBlock and endBlock
    rewriter.inlineRegionBefore(loop.region(), endBlock);

    // Get loop values from the LoopOp
    mlir::Value low = loop.lowerBound();
    mlir::Value high = loop.upperBound();
    mlir::Value step = loop.step();
    if (!low || !high) {
      return failure();
    }


    // Initalization block
    rewriter.setInsertionPointToEnd(initBlock);
    mlir::Value zero = rewriter.create<mlir::ConstantIndexOp>(loc, 0);
    mlir::Value one = rewriter.create<mlir::ConstantIndexOp>(loc, 1);
    mlir::Value diff = rewriter.create<mlir::SubIOp>(loc, high, low);
    mlir::Value distance = rewriter.create<mlir::AddIOp>(loc, diff, step);
    mlir::Value normalizedDistance =
      rewriter.create<mlir::SignedDivIOp>(loc, distance, step);
    mlir::Type storeType = normalizedDistance.getType();
    mlir::Value tripCounter =
      rewriter.create<fir::AllocaOp>(loc, storeType, llvm::None);
    rewriter.create<fir::StoreOp>(loc, normalizedDistance, tripCounter);

    SmallVector<mlir::Value, 8> loopOperands;
    loopOperands.push_back(low);
    mlir::ValueRange operands = loop.getIterOperands();
    loopOperands.append(operands.begin(), operands.end());

    // TODO: replace with a command line flag
    // onetrip flag determines whether loop should be executed once, before
    // conditionals are checked
    static const bool onetrip = false;
    if (onetrip) {
      rewriter.create<BranchOp>(loc, firstBlock, ArrayRef<Value>());
    } else {
      rewriter.create<BranchOp>(loc, conditionalBlock, loopOperands);
    }


    // Last loop block
    mlir::Operation *terminator = lastBlock->getTerminator();
    rewriter.setInsertionPointToEnd(lastBlock);
    mlir::Value index = conditionalBlock->getArgument(0);
    mlir::Value steppedIndex =
      rewriter.create<AddIOp>(loc, index, step).getResult();
    mlir::Value tripCount = rewriter.create<fir::LoadOp>(loc, tripCounter);
    mlir::Value steppedTripCount =
      rewriter.create<mlir::SubIOp>(loc, tripCount, one);
    rewriter.create<fir::StoreOp>(loc, steppedTripCount, tripCounter);

    SmallVector<mlir::Value, 8> loopCarried;
    loopCarried.push_back(steppedIndex);
    loopCarried.append(terminator->operand_begin(), terminator->operand_end());
    rewriter.create<BranchOp>(loc, conditionalBlock, loopCarried);
    rewriter.eraseOp(terminator);

    if (!steppedIndex) {
      return failure();
    }


    // Conditional block
    rewriter.setInsertionPointToEnd(conditionalBlock);
    mlir::Value tripCountValue =
      rewriter.create<fir::LoadOp>(loc, tripCounter);
    mlir::Value comparison =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt, tripCountValue, zero);

    rewriter.create<CondBranchOp>(loc, comparison, firstBlock,
                                  ArrayRef<Value>(), endBlock,
                                  ArrayRef<Value>());


    // The result of the loop operation is the values of the condition block
    // arguments except the induction variable on the last iteration.
    rewriter.replaceOp(loop, conditionalBlock->getArguments().drop_front());

    return success();
  }
};


// Conversion to the SCF dialect.
//
// FIR loops that cannot be converted to the affine dialect will remain as
// `fir.do_loop` operations.  These can be converted to `scf.for` operations.
// MLIR includes a pass to lower `scf.for` operations to a CFG.

/// Convert `fir.if` to `scf.if`
class ScfIfConv : public mlir::OpRewritePattern<fir::WhereOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::ResultOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op);
    return success();
  }
};

/// Convert `fir.if` to control-flow
class ScfIfConv : public mlir::OpRewritePattern<fir::WhereOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(WhereOp where,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = where.getLoc();

    // Split the block containing the 'fir.if' into two parts.  The part before
    // will contain the condition, the part after will be the continuation
    // point.
    auto *condBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
    mlir::Block *continueBlock;
    if (where.getNumResults() == 0) {
      continueBlock = remainingOpsBlock;
    } else {
      continueBlock =
          rewriter.createBlock(remainingOpsBlock, where.getResultTypes());
      rewriter.create<BranchOp>(loc, remainingOpsBlock);
    }

    // Move blocks from the "then" region to the region containing 'fir.if',
    // place it before the continuation block, and branch to it.
    auto &whereRegion = where.whereRegion();
    auto *whereBlock = &whereRegion.front();
    mlir::Operation *whereTerminator = whereRegion.back().getTerminator();
    mlir::ValueRange whereTerminatorOperands = whereTerminator->getOperands();
    rewriter.setInsertionPointToEnd(&whereRegion.back());
    rewriter.create<BranchOp>(loc, continueBlock, whereTerminatorOperands);
    rewriter.eraseOp(whereTerminator);
    rewriter.inlineRegionBefore(whereRegion, continueBlock);

    // Move blocks from the "else" region (if present) to the region containing
    // 'fir.if', place it before the continuation block and branch to it.  It
    // will be placed after the "then" regions.
    auto *otherwiseBlock = continueBlock;
    auto &otherwiseRegion = where.otherRegion();
    if (!otherwiseRegion.empty()) {
      otherwiseBlock = &otherwiseRegion.front();
      mlir::Operation *otherwiseTerm = otherwiseRegion.back().getTerminator();
      mlir::ValueRange otherwiseTermOperands = otherwiseTerm->getOperands();
      rewriter.setInsertionPointToEnd(&otherwiseRegion.back());
      rewriter.create<BranchOp>(loc, continueBlock, otherwiseTermOperands);
      rewriter.eraseOp(otherwiseTerm);
      rewriter.inlineRegionBefore(otherwiseRegion, continueBlock);
    }

    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<mlir::CondBranchOp>(
        loc, where.condition(), whereBlock, llvm::ArrayRef<mlir::Value>(),
        otherwiseBlock, llvm::ArrayRef<mlir::Value>());
    rewriter.replaceOp(where, continueBlock->getArguments());
    return success();
  }
};

/// Convert `fir.iter_while` to control-flow.
class ScfIterWhileConv : public mlir::OpRewritePattern<fir::IterWhileOp> {
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
                                  llvm::ArrayRef<mlir::Value>(), endBlock,
                                  llvm::ArrayRef<mlir::Value>());
    // The result of the loop operation is the values of the condition block
    // arguments except the induction variable on the last iteration.
    rewriter.replaceOp(whileOp, conditionBlock->getArguments().drop_front());
    return success();
  }
};

/// Convert FIR structured control flow ops to SCF ops.
class ScfDialectConversion
    : public mlir::PassWrapper<ScfDialectConversion, mlir::FunctionPass> {
public:
  void runOnFunction() override {
    if (disableScfConversion)
      return;

    auto *context = &getContext();
    mlir::OwningRewritePatternList patterns1;
    patterns1.insert<ScfIterWhileConv>(context);

    mlir::OwningRewritePatternList patterns2;
    patterns2.insert<ScfLoopConv, ScfIfConv, ScfResultConv>(context);
    mlir::ConversionTarget target = *context;
    target.addLegalDialect<mlir::AffineDialect, FIROpsDialect,
                           mlir::scf::SCFDialect, mlir::StandardOpsDialect>();

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
                      "error in converting to scf dialect\n");
      signalPassFailure();
    }
  }
};
} // namespace

/// Convert FIR's structured control flow ops to SCF ops.  This
/// conversion enables the `createLowerToCFGPass` to transform these to CFG
/// form.
std::unique_ptr<mlir::Pass> fir::createLowerToScfPass() {
  return std::make_unique<ScfDialectConversion>();
}

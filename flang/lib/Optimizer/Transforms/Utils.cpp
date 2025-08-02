//===-- Utils.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/Utils.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

/// Convert fir::DoLoopOp to control-flow operations
std::pair<mlir::Block *, mlir::Block *>
fir::convertDoLoopToCFG(DoLoopOp loop, mlir::PatternRewriter &rewriter,
                        bool setNSW, bool forceLoopToExecuteOnce) {
  auto loc = loop.getLoc();
  mlir::arith::IntegerOverflowFlags flags{};
  if (setNSW)
    flags = bitEnumSet(flags, mlir::arith::IntegerOverflowFlags::nsw);
  auto iofAttr =
      mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), flags);

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
  auto diff = mlir::arith::SubIOp::create(rewriter, loc, high, low);
  auto distance = mlir::arith::AddIOp::create(rewriter, loc, diff, step);
  mlir::Value iters =
      mlir::arith::DivSIOp::create(rewriter, loc, distance, step);

  if (forceLoopToExecuteOnce) {
    auto zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto cond = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::sle, iters, zero);
    auto one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    iters = mlir::arith::SelectOp::create(rewriter, loc, cond, one, iters);
  }

  llvm::SmallVector<mlir::Value> loopOperands;
  loopOperands.push_back(low);
  auto operands = loop.getIterOperands();
  loopOperands.append(operands.begin(), operands.end());
  loopOperands.push_back(iters);

  mlir::cf::BranchOp::create(rewriter, loc, conditionalBlock, loopOperands);

  // Last loop block
  auto *terminator = lastBlock->getTerminator();
  rewriter.setInsertionPointToEnd(lastBlock);
  auto iv = conditionalBlock->getArgument(0);
  mlir::Value steppedIndex =
      mlir::arith::AddIOp::create(rewriter, loc, iv, step, iofAttr);
  assert(steppedIndex && "must be a Value");
  auto lastArg = conditionalBlock->getNumArguments() - 1;
  auto itersLeft = conditionalBlock->getArgument(lastArg);
  auto one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
  mlir::Value itersMinusOne =
      mlir::arith::SubIOp::create(rewriter, loc, itersLeft, one);

  llvm::SmallVector<mlir::Value> loopCarried;
  loopCarried.push_back(steppedIndex);
  auto begin = loop.getFinalValue() ? std::next(terminator->operand_begin())
                                    : terminator->operand_begin();
  loopCarried.append(begin, terminator->operand_end());
  loopCarried.push_back(itersMinusOne);
  auto backEdge =
      mlir::cf::BranchOp::create(rewriter, loc, conditionalBlock, loopCarried);
  rewriter.eraseOp(terminator);

  // Copy loop annotations from the do loop to the loop back edge.
  if (auto ann = loop.getLoopAnnotation())
    backEdge->setAttr("loop_annotation", *ann);

  // Conditional block
  rewriter.setInsertionPointToEnd(conditionalBlock);
  auto zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto comparison = mlir::arith::CmpIOp::create(
      rewriter, loc, mlir::arith::CmpIPredicate::sgt, itersLeft, zero);

  mlir::cf::CondBranchOp::create(rewriter, loc, comparison, firstBlock,
                                 llvm::ArrayRef<mlir::Value>(), endBlock,
                                 llvm::ArrayRef<mlir::Value>());

  // The result of the loop operation is the values of the condition block
  // arguments except the induction variable on the last iteration.
  auto args = loop.getFinalValue()
                  ? conditionalBlock->getArguments()
                  : conditionalBlock->getArguments().drop_front();
  rewriter.replaceOp(loop, args.drop_back());

  return std::make_pair(conditionalBlock, lastBlock);
}

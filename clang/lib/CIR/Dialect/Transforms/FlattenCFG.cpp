//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements pass that inlines CIR operations regions into the parent
// function region.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/MissingFeatures.h"

using namespace mlir;
using namespace cir;

namespace {

/// Lowers operations with the terminator trait that have a single successor.
void lowerTerminator(mlir::Operation *op, mlir::Block *dest,
                     mlir::PatternRewriter &rewriter) {
  assert(op->hasTrait<mlir::OpTrait::IsTerminator>() && "not a terminator");
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<cir::BrOp>(op, dest);
}

/// Walks a region while skipping operations of type `Ops`. This ensures the
/// callback is not applied to said operations and its children.
template <typename... Ops>
void walkRegionSkipping(
    mlir::Region &region,
    mlir::function_ref<mlir::WalkResult(mlir::Operation *)> callback) {
  region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (isa<Ops...>(op))
      return mlir::WalkResult::skip();
    return callback(op);
  });
}

struct CIRFlattenCFGPass : public CIRFlattenCFGBase<CIRFlattenCFGPass> {

  CIRFlattenCFGPass() = default;
  void runOnOperation() override;
};

struct CIRIfFlattening : public mlir::OpRewritePattern<cir::IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cir::IfOp ifOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    mlir::Location loc = ifOp.getLoc();
    bool emptyElse = ifOp.getElseRegion().empty();
    mlir::Block *currentBlock = rewriter.getInsertionBlock();
    mlir::Block *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *continueBlock;
    if (ifOp->getResults().empty())
      continueBlock = remainingOpsBlock;
    else
      llvm_unreachable("NYI");

    // Inline the region
    mlir::Block *thenBeforeBody = &ifOp.getThenRegion().front();
    mlir::Block *thenAfterBody = &ifOp.getThenRegion().back();
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), continueBlock);

    rewriter.setInsertionPointToEnd(thenAfterBody);
    if (auto thenYieldOp =
            dyn_cast<cir::YieldOp>(thenAfterBody->getTerminator())) {
      rewriter.replaceOpWithNewOp<cir::BrOp>(thenYieldOp, thenYieldOp.getArgs(),
                                             continueBlock);
    }

    rewriter.setInsertionPointToEnd(continueBlock);

    // Has else region: inline it.
    mlir::Block *elseBeforeBody = nullptr;
    mlir::Block *elseAfterBody = nullptr;
    if (!emptyElse) {
      elseBeforeBody = &ifOp.getElseRegion().front();
      elseAfterBody = &ifOp.getElseRegion().back();
      rewriter.inlineRegionBefore(ifOp.getElseRegion(), continueBlock);
    } else {
      elseBeforeBody = elseAfterBody = continueBlock;
    }

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<cir::BrCondOp>(loc, ifOp.getCondition(), thenBeforeBody,
                                   elseBeforeBody);

    if (!emptyElse) {
      rewriter.setInsertionPointToEnd(elseAfterBody);
      if (auto elseYieldOP =
              dyn_cast<cir::YieldOp>(elseAfterBody->getTerminator())) {
        rewriter.replaceOpWithNewOp<cir::BrOp>(
            elseYieldOP, elseYieldOP.getArgs(), continueBlock);
      }
    }

    rewriter.replaceOp(ifOp, continueBlock->getArguments());
    return mlir::success();
  }
};

class CIRScopeOpFlattening : public mlir::OpRewritePattern<cir::ScopeOp> {
public:
  using OpRewritePattern<cir::ScopeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ScopeOp scopeOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    mlir::Location loc = scopeOp.getLoc();

    // Empty scope: just remove it.
    // TODO: Remove this logic once CIR uses MLIR infrastructure to remove
    // trivially dead operations. MLIR canonicalizer is too aggressive and we
    // need to either (a) make sure all our ops model all side-effects and/or
    // (b) have more options in the canonicalizer in MLIR to temper
    // aggressiveness level.
    if (scopeOp.isEmpty()) {
      rewriter.eraseOp(scopeOp);
      return mlir::success();
    }

    // Split the current block before the ScopeOp to create the inlining
    // point.
    mlir::Block *currentBlock = rewriter.getInsertionBlock();
    mlir::Block *continueBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    if (scopeOp.getNumResults() > 0)
      continueBlock->addArguments(scopeOp.getResultTypes(), loc);

    // Inline body region.
    mlir::Block *beforeBody = &scopeOp.getScopeRegion().front();
    mlir::Block *afterBody = &scopeOp.getScopeRegion().back();
    rewriter.inlineRegionBefore(scopeOp.getScopeRegion(), continueBlock);

    // Save stack and then branch into the body of the region.
    rewriter.setInsertionPointToEnd(currentBlock);
    assert(!cir::MissingFeatures::stackSaveOp());
    rewriter.create<cir::BrOp>(loc, mlir::ValueRange(), beforeBody);

    // Replace the scopeop return with a branch that jumps out of the body.
    // Stack restore before leaving the body region.
    rewriter.setInsertionPointToEnd(afterBody);
    if (auto yieldOp = dyn_cast<cir::YieldOp>(afterBody->getTerminator())) {
      rewriter.replaceOpWithNewOp<cir::BrOp>(yieldOp, yieldOp.getArgs(),
                                             continueBlock);
    }

    // Replace the op with values return from the body region.
    rewriter.replaceOp(scopeOp, continueBlock->getArguments());

    return mlir::success();
  }
};

class CIRSwitchOpFlattening : public mlir::OpRewritePattern<cir::SwitchOp> {
public:
  using OpRewritePattern<cir::SwitchOp>::OpRewritePattern;

  inline void rewriteYieldOp(mlir::PatternRewriter &rewriter,
                             cir::YieldOp yieldOp,
                             mlir::Block *destination) const {
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<cir::BrOp>(yieldOp, yieldOp.getOperands(),
                                           destination);
  }

  // Return the new defaultDestination block.
  Block *condBrToRangeDestination(cir::SwitchOp op,
                                  mlir::PatternRewriter &rewriter,
                                  mlir::Block *rangeDestination,
                                  mlir::Block *defaultDestination,
                                  const APInt &lowerBound,
                                  const APInt &upperBound) const {
    assert(lowerBound.sle(upperBound) && "Invalid range");
    mlir::Block *resBlock = rewriter.createBlock(defaultDestination);
    cir::IntType sIntType = cir::IntType::get(op.getContext(), 32, true);
    cir::IntType uIntType = cir::IntType::get(op.getContext(), 32, false);

    cir::ConstantOp rangeLength = rewriter.create<cir::ConstantOp>(
        op.getLoc(), cir::IntAttr::get(sIntType, upperBound - lowerBound));

    cir::ConstantOp lowerBoundValue = rewriter.create<cir::ConstantOp>(
        op.getLoc(), cir::IntAttr::get(sIntType, lowerBound));
    cir::BinOp diffValue =
        rewriter.create<cir::BinOp>(op.getLoc(), sIntType, cir::BinOpKind::Sub,
                                    op.getCondition(), lowerBoundValue);

    // Use unsigned comparison to check if the condition is in the range.
    cir::CastOp uDiffValue = rewriter.create<cir::CastOp>(
        op.getLoc(), uIntType, CastKind::integral, diffValue);
    cir::CastOp uRangeLength = rewriter.create<cir::CastOp>(
        op.getLoc(), uIntType, CastKind::integral, rangeLength);

    cir::CmpOp cmpResult = rewriter.create<cir::CmpOp>(
        op.getLoc(), cir::BoolType::get(op.getContext()), cir::CmpOpKind::le,
        uDiffValue, uRangeLength);
    rewriter.create<cir::BrCondOp>(op.getLoc(), cmpResult, rangeDestination,
                                   defaultDestination);
    return resBlock;
  }

  mlir::LogicalResult
  matchAndRewrite(cir::SwitchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    llvm::SmallVector<CaseOp> cases;
    op.collectCases(cases);

    // Empty switch statement: just erase it.
    if (cases.empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // Create exit block from the next node of cir.switch op.
    mlir::Block *exitBlock = rewriter.splitBlock(
        rewriter.getBlock(), op->getNextNode()->getIterator());

    // We lower cir.switch op in the following process:
    // 1. Inline the region from the switch op after switch op.
    // 2. Traverse each cir.case op:
    //    a. Record the entry block, block arguments and condition for every
    //    case. b. Inline the case region after the case op.
    // 3. Replace the empty cir.switch.op with the new cir.switchflat op by the
    //    recorded block and conditions.

    // inline everything from switch body between the switch op and the exit
    // block.
    {
      cir::YieldOp switchYield = nullptr;
      // Clear switch operation.
      for (mlir::Block &block :
           llvm::make_early_inc_range(op.getBody().getBlocks()))
        if (auto yieldOp = dyn_cast<cir::YieldOp>(block.getTerminator()))
          switchYield = yieldOp;

      assert(!op.getBody().empty());
      mlir::Block *originalBlock = op->getBlock();
      mlir::Block *swopBlock =
          rewriter.splitBlock(originalBlock, op->getIterator());
      rewriter.inlineRegionBefore(op.getBody(), exitBlock);

      if (switchYield)
        rewriteYieldOp(rewriter, switchYield, exitBlock);

      rewriter.setInsertionPointToEnd(originalBlock);
      rewriter.create<cir::BrOp>(op.getLoc(), swopBlock);
    }

    // Allocate required data structures (disconsider default case in
    // vectors).
    llvm::SmallVector<mlir::APInt, 8> caseValues;
    llvm::SmallVector<mlir::Block *, 8> caseDestinations;
    llvm::SmallVector<mlir::ValueRange, 8> caseOperands;

    llvm::SmallVector<std::pair<APInt, APInt>> rangeValues;
    llvm::SmallVector<mlir::Block *> rangeDestinations;
    llvm::SmallVector<mlir::ValueRange> rangeOperands;

    // Initialize default case as optional.
    mlir::Block *defaultDestination = exitBlock;
    mlir::ValueRange defaultOperands = exitBlock->getArguments();

    // Digest the case statements values and bodies.
    for (cir::CaseOp caseOp : cases) {
      mlir::Region &region = caseOp.getCaseRegion();

      // Found default case: save destination and operands.
      switch (caseOp.getKind()) {
      case cir::CaseOpKind::Default:
        defaultDestination = &region.front();
        defaultOperands = defaultDestination->getArguments();
        break;
      case cir::CaseOpKind::Range:
        assert(caseOp.getValue().size() == 2 &&
               "Case range should have 2 case value");
        rangeValues.push_back(
            {cast<cir::IntAttr>(caseOp.getValue()[0]).getValue(),
             cast<cir::IntAttr>(caseOp.getValue()[1]).getValue()});
        rangeDestinations.push_back(&region.front());
        rangeOperands.push_back(rangeDestinations.back()->getArguments());
        break;
      case cir::CaseOpKind::Anyof:
      case cir::CaseOpKind::Equal:
        // AnyOf cases kind can have multiple values, hence the loop below.
        for (const mlir::Attribute &value : caseOp.getValue()) {
          caseValues.push_back(cast<cir::IntAttr>(value).getValue());
          caseDestinations.push_back(&region.front());
          caseOperands.push_back(caseDestinations.back()->getArguments());
        }
        break;
      }

      // Handle break statements.
      walkRegionSkipping<cir::LoopOpInterface, cir::SwitchOp>(
          region, [&](mlir::Operation *op) {
            if (!isa<cir::BreakOp>(op))
              return mlir::WalkResult::advance();

            lowerTerminator(op, exitBlock, rewriter);
            return mlir::WalkResult::skip();
          });

      // Track fallthrough in cases.
      for (mlir::Block &blk : region.getBlocks()) {
        if (blk.getNumSuccessors())
          continue;

        if (auto yieldOp = dyn_cast<cir::YieldOp>(blk.getTerminator())) {
          mlir::Operation *nextOp = caseOp->getNextNode();
          assert(nextOp && "caseOp is not expected to be the last op");
          mlir::Block *oldBlock = nextOp->getBlock();
          mlir::Block *newBlock =
              rewriter.splitBlock(oldBlock, nextOp->getIterator());
          rewriter.setInsertionPointToEnd(oldBlock);
          rewriter.create<cir::BrOp>(nextOp->getLoc(), mlir::ValueRange(),
                                     newBlock);
          rewriteYieldOp(rewriter, yieldOp, newBlock);
        }
      }

      mlir::Block *oldBlock = caseOp->getBlock();
      mlir::Block *newBlock =
          rewriter.splitBlock(oldBlock, caseOp->getIterator());

      mlir::Block &entryBlock = caseOp.getCaseRegion().front();
      rewriter.inlineRegionBefore(caseOp.getCaseRegion(), newBlock);

      // Create a branch to the entry of the inlined region.
      rewriter.setInsertionPointToEnd(oldBlock);
      rewriter.create<cir::BrOp>(caseOp.getLoc(), &entryBlock);
    }

    // Remove all cases since we've inlined the regions.
    for (cir::CaseOp caseOp : cases) {
      mlir::Block *caseBlock = caseOp->getBlock();
      // Erase the block with no predecessors here to make the generated code
      // simpler a little bit.
      if (caseBlock->hasNoPredecessors())
        rewriter.eraseBlock(caseBlock);
      else
        rewriter.eraseOp(caseOp);
    }

    for (auto [rangeVal, operand, destination] :
         llvm::zip(rangeValues, rangeOperands, rangeDestinations)) {
      APInt lowerBound = rangeVal.first;
      APInt upperBound = rangeVal.second;

      // The case range is unreachable, skip it.
      if (lowerBound.sgt(upperBound))
        continue;

      // If range is small, add multiple switch instruction cases.
      // This magical number is from the original CGStmt code.
      constexpr int kSmallRangeThreshold = 64;
      if ((upperBound - lowerBound)
              .ult(llvm::APInt(32, kSmallRangeThreshold))) {
        for (APInt iValue = lowerBound; iValue.sle(upperBound); ++iValue) {
          caseValues.push_back(iValue);
          caseOperands.push_back(operand);
          caseDestinations.push_back(destination);
        }
        continue;
      }

      defaultDestination =
          condBrToRangeDestination(op, rewriter, destination,
                                   defaultDestination, lowerBound, upperBound);
      defaultOperands = operand;
    }

    // Set switch op to branch to the newly created blocks.
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<cir::SwitchFlatOp>(
        op, op.getCondition(), defaultDestination, defaultOperands, caseValues,
        caseDestinations, caseOperands);

    return mlir::success();
  }
};

class CIRLoopOpInterfaceFlattening
    : public mlir::OpInterfaceRewritePattern<cir::LoopOpInterface> {
public:
  using mlir::OpInterfaceRewritePattern<
      cir::LoopOpInterface>::OpInterfaceRewritePattern;

  inline void lowerConditionOp(cir::ConditionOp op, mlir::Block *body,
                               mlir::Block *exit,
                               mlir::PatternRewriter &rewriter) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<cir::BrCondOp>(op, op.getCondition(), body,
                                               exit);
  }

  mlir::LogicalResult
  matchAndRewrite(cir::LoopOpInterface op,
                  mlir::PatternRewriter &rewriter) const final {
    // Setup CFG blocks.
    mlir::Block *entry = rewriter.getInsertionBlock();
    mlir::Block *exit =
        rewriter.splitBlock(entry, rewriter.getInsertionPoint());
    mlir::Block *cond = &op.getCond().front();
    mlir::Block *body = &op.getBody().front();
    mlir::Block *step =
        (op.maybeGetStep() ? &op.maybeGetStep()->front() : nullptr);

    // Setup loop entry branch.
    rewriter.setInsertionPointToEnd(entry);
    rewriter.create<cir::BrOp>(op.getLoc(), &op.getEntry().front());

    // Branch from condition region to body or exit.
    auto conditionOp = cast<cir::ConditionOp>(cond->getTerminator());
    lowerConditionOp(conditionOp, body, exit, rewriter);

    // TODO(cir): Remove the walks below. It visits operations unnecessarily.
    // However, to solve this we would likely need a custom DialectConversion
    // driver to customize the order that operations are visited.

    // Lower continue statements.
    mlir::Block *dest = (step ? step : cond);
    op.walkBodySkippingNestedLoops([&](mlir::Operation *op) {
      if (!isa<cir::ContinueOp>(op))
        return mlir::WalkResult::advance();

      lowerTerminator(op, dest, rewriter);
      return mlir::WalkResult::skip();
    });

    // Lower break statements.
    assert(!cir::MissingFeatures::switchOp());
    walkRegionSkipping<cir::LoopOpInterface>(
        op.getBody(), [&](mlir::Operation *op) {
          if (!isa<cir::BreakOp>(op))
            return mlir::WalkResult::advance();

          lowerTerminator(op, exit, rewriter);
          return mlir::WalkResult::skip();
        });

    // Lower optional body region yield.
    for (mlir::Block &blk : op.getBody().getBlocks()) {
      auto bodyYield = dyn_cast<cir::YieldOp>(blk.getTerminator());
      if (bodyYield)
        lowerTerminator(bodyYield, (step ? step : cond), rewriter);
    }

    // Lower mandatory step region yield.
    if (step)
      lowerTerminator(cast<cir::YieldOp>(step->getTerminator()), cond,
                      rewriter);

    // Move region contents out of the loop op.
    rewriter.inlineRegionBefore(op.getCond(), exit);
    rewriter.inlineRegionBefore(op.getBody(), exit);
    if (step)
      rewriter.inlineRegionBefore(*op.maybeGetStep(), exit);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CIRTernaryOpFlattening : public mlir::OpRewritePattern<cir::TernaryOp> {
public:
  using OpRewritePattern<cir::TernaryOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cir::TernaryOp op,
                  mlir::PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Block *condBlock = rewriter.getInsertionBlock();
    Block::iterator opPosition = rewriter.getInsertionPoint();
    Block *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
    llvm::SmallVector<mlir::Location, 2> locs;
    // Ternary result is optional, make sure to populate the location only
    // when relevant.
    if (op->getResultTypes().size())
      locs.push_back(loc);
    Block *continueBlock =
        rewriter.createBlock(remainingOpsBlock, op->getResultTypes(), locs);
    rewriter.create<cir::BrOp>(loc, remainingOpsBlock);

    Region &trueRegion = op.getTrueRegion();
    Block *trueBlock = &trueRegion.front();
    mlir::Operation *trueTerminator = trueRegion.back().getTerminator();
    rewriter.setInsertionPointToEnd(&trueRegion.back());

    // Handle both yield and unreachable terminators (throw expressions)
    if (auto trueYieldOp = dyn_cast<cir::YieldOp>(trueTerminator)) {
      rewriter.replaceOpWithNewOp<cir::BrOp>(trueYieldOp, trueYieldOp.getArgs(),
                                             continueBlock);
    } else if (isa<cir::UnreachableOp>(trueTerminator)) {
      // Terminator is unreachable (e.g., from throw), just keep it
    } else {
      trueTerminator->emitError("unexpected terminator in ternary true region, "
                                "expected yield or unreachable, got: ")
          << trueTerminator->getName();
      return mlir::failure();
    }
    rewriter.inlineRegionBefore(trueRegion, continueBlock);

    Block *falseBlock = continueBlock;
    Region &falseRegion = op.getFalseRegion();

    falseBlock = &falseRegion.front();
    mlir::Operation *falseTerminator = falseRegion.back().getTerminator();
    rewriter.setInsertionPointToEnd(&falseRegion.back());

    // Handle both yield and unreachable terminators (throw expressions)
    if (auto falseYieldOp = dyn_cast<cir::YieldOp>(falseTerminator)) {
      rewriter.replaceOpWithNewOp<cir::BrOp>(
          falseYieldOp, falseYieldOp.getArgs(), continueBlock);
    } else if (isa<cir::UnreachableOp>(falseTerminator)) {
      // Terminator is unreachable (e.g., from throw), just keep it
    } else {
      falseTerminator->emitError("unexpected terminator in ternary false "
                                 "region, expected yield or unreachable, got: ")
          << falseTerminator->getName();
      return mlir::failure();
    }
    rewriter.inlineRegionBefore(falseRegion, continueBlock);

    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<cir::BrCondOp>(loc, op.getCond(), trueBlock, falseBlock);

    rewriter.replaceOp(op, continueBlock->getArguments());

    // Ok, we're done!
    return mlir::success();
  }
};

void populateFlattenCFGPatterns(RewritePatternSet &patterns) {
  patterns
      .add<CIRIfFlattening, CIRLoopOpInterfaceFlattening, CIRScopeOpFlattening,
           CIRSwitchOpFlattening, CIRTernaryOpFlattening>(
          patterns.getContext());
}

void CIRFlattenCFGPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateFlattenCFGPatterns(patterns);

  // Collect operations to apply patterns.
  llvm::SmallVector<Operation *, 16> ops;
  getOperation()->walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
    assert(!cir::MissingFeatures::ifOp());
    assert(!cir::MissingFeatures::switchOp());
    assert(!cir::MissingFeatures::tryOp());
    if (isa<IfOp, ScopeOp, SwitchOp, LoopOpInterface, TernaryOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (applyOpPatternsGreedily(ops, std::move(patterns)).failed())
    signalPassFailure();
}

} // namespace

namespace mlir {

std::unique_ptr<Pass> createCIRFlattenCFGPass() {
  return std::make_unique<CIRFlattenCFGPass>();
}

} // namespace mlir

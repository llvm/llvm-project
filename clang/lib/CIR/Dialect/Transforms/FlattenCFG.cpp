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
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/CIRTransformUtils.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CIRFLATTENCFG
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

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

/// Check whether a region contains any nested op with regions (i.e. structured
/// CIR ops that must be flattened before their parent). The greedy pattern
/// rewriter doesn't guarantee inside-out processing order — when a pattern
/// fires and modifies IR, newly created ops go onto the worklist and can be
/// visited in any order. So each flattening pattern must explicitly defer
/// until its nested structured ops are flat.
///
/// CaseOps are excluded because they are structural children of SwitchOp and
/// are handled by the SwitchOp flattening pattern.
static bool hasNestedOpsToFlatten(mlir::Region &region) {
  return region
      .walk([](mlir::Operation *op) {
        if (op->getNumRegions() > 0 && !isa<cir::CaseOp>(op))
          return mlir::WalkResult::interrupt();
        return mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

/// True if `op` is a non-returning terminator — currently `cir.unreachable`
/// or `cir.trap`. Such terminators don't fall through and don't yield a
/// value, so when flattening a region they can be left in place rather than
/// being replaced with a branch to the continuation block. Add new ops here
/// (e.g. a hypothetical `cir.abort`) so every flattening pattern picks them
/// up at once.
static bool isNonReturningTerminator(mlir::Operation *op) {
  return mlir::isa_and_nonnull<cir::UnreachableOp, cir::TrapOp>(op);
}

/// Rewrite the terminator of `region`'s exit block so that, after
/// flattening, control falls through to `continueBlock`. The exit
/// terminator is expected to be either:
///   - `cir.yield`: replaced with `cir.br` to `continueBlock` (yielded
///     args become the destination block's arguments).
///   - non-returning (`cir.unreachable`, `cir.trap`): left in place — no
///     branch is needed.
///
/// On success returns `success()`. If the terminator is anything else, an
/// error is emitted and `failure()` is returned. NOTE: callers in this
/// file have typically already mutated IR (splitBlock / createBlock) by
/// the time this is invoked, so the MLIR pattern rewriter contract
/// requires them to still return `success()` from the surrounding
/// pattern; the `failure()` here just signals "stop trying to wire up
/// this region".
static mlir::LogicalResult
rewriteRegionExitToContinue(mlir::PatternRewriter &rewriter,
                            mlir::Region &region, mlir::Block *continueBlock,
                            llvm::StringRef regionDescription) {
  mlir::Operation *terminator = region.back().getTerminator();
  rewriter.setInsertionPointToEnd(&region.back());
  if (auto yieldOp = mlir::dyn_cast<cir::YieldOp>(terminator)) {
    rewriter.replaceOpWithNewOp<cir::BrOp>(yieldOp, yieldOp.getArgs(),
                                           continueBlock);
    return mlir::success();
  }
  if (isNonReturningTerminator(terminator))
    return mlir::success();
  terminator->emitError("unexpected terminator in ")
      << regionDescription
      << " region, expected yield, unreachable, or trap, got: "
      << terminator->getName();
  return mlir::failure();
}

struct CIRFlattenCFGPass : public impl::CIRFlattenCFGBase<CIRFlattenCFGPass> {

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
    cir::BrCondOp::create(rewriter, loc, ifOp.getCondition(), thenBeforeBody,
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
    cir::BrOp::create(rewriter, loc, mlir::ValueRange(), beforeBody);

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

    cir::ConstantOp rangeLength = cir::ConstantOp::create(
        rewriter, op.getLoc(),
        cir::IntAttr::get(sIntType, upperBound - lowerBound));

    cir::ConstantOp lowerBoundValue = cir::ConstantOp::create(
        rewriter, op.getLoc(), cir::IntAttr::get(sIntType, lowerBound));
    mlir::Value diffValue = cir::SubOp::create(
        rewriter, op.getLoc(), op.getCondition(), lowerBoundValue);

    // Use unsigned comparison to check if the condition is in the range.
    cir::CastOp uDiffValue = cir::CastOp::create(
        rewriter, op.getLoc(), uIntType, CastKind::integral, diffValue);
    cir::CastOp uRangeLength = cir::CastOp::create(
        rewriter, op.getLoc(), uIntType, CastKind::integral, rangeLength);

    cir::CmpOp cmpResult = cir::CmpOp::create(
        rewriter, op.getLoc(), cir::CmpOpKind::le, uDiffValue, uRangeLength);
    cir::BrCondOp::create(rewriter, op.getLoc(), cmpResult, rangeDestination,
                          defaultDestination);
    return resBlock;
  }

  mlir::LogicalResult
  matchAndRewrite(cir::SwitchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // All nested structured CIR ops must be flattened before the switch.
    // Break statements inside nested structured ops would create branches to
    // blocks outside those ops' regions, which is invalid. Fail the match so
    // the pattern rewriter will process them first.
    for (mlir::Region &region : op->getRegions())
      if (hasNestedOpsToFlatten(region))
        return mlir::failure();

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
      cir::BrOp::create(rewriter, op.getLoc(), swopBlock);
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
          cir::BrOp::create(rewriter, nextOp->getLoc(), mlir::ValueRange(),
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
      cir::BrOp::create(rewriter, caseOp.getLoc(), &entryBlock);
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
    // All nested structured CIR ops must be flattened before the loop.
    // Break/continue statements inside nested structured ops would create
    // branches to blocks outside those ops' regions, which is invalid. Fail
    // the match so the pattern rewriter will process them first.
    for (mlir::Region &region : op->getRegions())
      if (hasNestedOpsToFlatten(region))
        return mlir::failure();

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
    cir::BrOp::create(rewriter, op.getLoc(), &op.getEntry().front());

    // Branch from condition region to body or exit. The ConditionOp may not
    // be in the first block of the condition region if a cleanup scope was
    // already flattened within it, introducing multiple blocks. The
    // ConditionOp is always the terminator of the last block.
    auto conditionOp =
        cast<cir::ConditionOp>(op.getCond().back().getTerminator());
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
    walkRegionSkipping<cir::LoopOpInterface, cir::SwitchOp>(
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

    // Lower mandatory step region yield. Like the condition region, the
    // YieldOp may be in the last block rather than the first if a cleanup
    // scope was already flattened within the step region.
    if (step)
      lowerTerminator(
          cast<cir::YieldOp>(op.maybeGetStep()->back().getTerminator()), cond,
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
    cir::BrOp::create(rewriter, loc, remainingOpsBlock);

    Region &trueRegion = op.getTrueRegion();
    Block *trueBlock = &trueRegion.front();
    // Wire up the true region's exit (cir.yield -> br, cir.unreachable /
    // cir.trap kept as-is). IR has already been modified by splitBlock /
    // createBlock above, so per the MLIR pattern rewriter contract we must
    // still return success() if the terminator turns out to be unexpected.
    if (failed(rewriteRegionExitToContinue(rewriter, trueRegion, continueBlock,
                                           "ternary true")))
      return mlir::success();
    rewriter.inlineRegionBefore(trueRegion, continueBlock);

    Block *falseBlock = continueBlock;
    Region &falseRegion = op.getFalseRegion();

    falseBlock = &falseRegion.front();
    if (failed(rewriteRegionExitToContinue(rewriter, falseRegion, continueBlock,
                                           "ternary false")))
      return mlir::success();
    rewriter.inlineRegionBefore(falseRegion, continueBlock);

    rewriter.setInsertionPointToEnd(condBlock);
    cir::BrCondOp::create(rewriter, loc, op.getCond(), trueBlock, falseBlock);

    rewriter.replaceOp(op, continueBlock->getArguments());

    // Ok, we're done!
    return mlir::success();
  }
};

// Get or create the cleanup destination slot for a function. This slot is
// shared across all cleanup scopes in the function to track which exit path
// to take after running cleanup code when there are multiple exits.
static cir::AllocaOp getOrCreateCleanupDestSlot(cir::FuncOp funcOp,
                                                mlir::PatternRewriter &rewriter,
                                                mlir::Location loc) {
  mlir::Block &entryBlock = funcOp.getBody().front();

  // Look for an existing cleanup dest slot in the entry block.
  auto it = llvm::find_if(entryBlock, [](auto &op) {
    return mlir::isa<AllocaOp>(&op) &&
           mlir::cast<AllocaOp>(&op).getCleanupDestSlot();
  });
  if (it != entryBlock.end())
    return mlir::cast<cir::AllocaOp>(*it);

  // Create a new cleanup dest slot at the start of the entry block.
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&entryBlock);
  cir::IntType s32Type =
      cir::IntType::get(rewriter.getContext(), 32, /*isSigned=*/true);
  cir::PointerType ptrToS32Type = cir::PointerType::get(s32Type);
  cir::CIRDataLayout dataLayout(funcOp->getParentOfType<mlir::ModuleOp>());
  uint64_t alignment = dataLayout.getAlignment(s32Type, true).value();
  auto allocaOp = cir::AllocaOp::create(
      rewriter, loc, ptrToS32Type, s32Type, "__cleanup_dest_slot",
      /*alignment=*/rewriter.getI64IntegerAttr(alignment));
  allocaOp.setCleanupDestSlot(true);
  return allocaOp;
}

/// Shared EH flattening utilities used by both CIRCleanupScopeOpFlattening
/// and CIRTryOpFlattening.

// Collect all function calls in a region that may throw exceptions and need
// to be replaced with try_call operations. Skips calls marked nothrow.
// Nested cleanup scopes and try ops are always flattened before their
// enclosing parents, so there are no nested regions to skip here.
static void
collectThrowingCalls(mlir::Region &region,
                     llvm::SmallVectorImpl<cir::CallOp> &callsToRewrite) {
  region.walk([&](cir::CallOp callOp) {
    if (!callOp.getNothrow())
      callsToRewrite.push_back(callOp);
  });
}

// Collect all cir.resume operations in a region that come from
// already-flattened try or cleanup scope operations. These resume ops need
// to be chained through this scope's EH handler instead of unwinding
// directly to the caller. Nested cleanup scopes and try ops are always
// flattened before their enclosing parents, so there are no nested regions
// to skip here.
static void collectResumeOps(mlir::Region &region,
                             llvm::SmallVectorImpl<cir::ResumeOp> &resumeOps) {
  region.walk([&](cir::ResumeOp resumeOp) { resumeOps.push_back(resumeOp); });
}

// Create a shared unwind destination block. The block contains a
// cir.eh.initiate operation (optionally with the cleanup attribute) and a
// branch to the given destination block, passing the eh_token.
static mlir::Block *buildUnwindBlock(mlir::Block *dest, bool isCleanupOnly,
                                     mlir::Location loc,
                                     mlir::Block *insertBefore,
                                     mlir::PatternRewriter &rewriter) {
  mlir::Block *unwindBlock = rewriter.createBlock(insertBefore);
  rewriter.setInsertionPointToEnd(unwindBlock);
  auto ehInitiate =
      cir::EhInitiateOp::create(rewriter, loc, /*cleanup=*/isCleanupOnly);
  cir::BrOp::create(rewriter, loc, mlir::ValueRange{ehInitiate.getEhToken()},
                    dest);
  return unwindBlock;
}

// Create a shared terminate unwind block for throwing calls in EH cleanup
// regions. When an exception is thrown during cleanup (unwinding), the C++
// standard requires that std::terminate() be called.
static mlir::Block *buildTerminateUnwindBlock(mlir::Location loc,
                                              mlir::Block *insertBefore,
                                              mlir::PatternRewriter &rewriter) {
  mlir::Block *terminateBlock = rewriter.createBlock(insertBefore);
  rewriter.setInsertionPointToEnd(terminateBlock);
  auto ehInitiate = cir::EhInitiateOp::create(rewriter, loc, /*cleanup=*/false);
  cir::EhTerminateOp::create(rewriter, loc, ehInitiate.getEhToken());
  return terminateBlock;
}

class CIRCleanupScopeOpFlattening
    : public mlir::OpRewritePattern<cir::CleanupScopeOp> {
public:
  using OpRewritePattern<cir::CleanupScopeOp>::OpRewritePattern;

  struct CleanupExit {
    // An operation that exits the cleanup scope (yield, break, continue,
    // return, etc.)
    mlir::Operation *exitOp;

    // A unique identifier for this exit's destination (used for switch dispatch
    // when there are multiple exits).
    int destinationId;

    CleanupExit(mlir::Operation *op, int id) : exitOp(op), destinationId(id) {}
  };

  // Determine whether a goto operation transfers control to a label that
  // exists somewhere inside the given region (or any of its nested regions).
  // Label names are unique within a function, so finding a matching cir.label
  // inside the region implies that the goto definitely targets that label and
  // therefore stays within the region. If no match is found, the goto either
  // exits the region or its target is unknown; in either case the caller must
  // treat it as exiting the region.
  static bool gotoTargetsLabelInRegion(cir::GotoOp gotoOp,
                                       mlir::Region &region) {
    llvm::StringRef targetLabel = gotoOp.getLabel();
    return region
        .walk([&](cir::LabelOp labelOp) {
          if (labelOp.getLabel() == targetLabel)
            return mlir::WalkResult::interrupt();
          return mlir::WalkResult::advance();
        })
        .wasInterrupted();
  }

  // Collect all operations that exit a cleanup scope body. Return, goto, break,
  // and continue can all require branches through the cleanup region. When a
  // loop is encountered, only return and goto are collected because break and
  // continue are handled by the loop and stay within the cleanup scope. When a
  // switch is encountered, return, goto and continue are collected because they
  // may all branch through the cleanup, but break is local to the switch. When
  // a nested cleanup scope is encountered, we recursively collect exits since
  // any return, goto, break, or continue from the nested cleanup will also
  // branch through the outer cleanup.
  //
  // A goto is only treated as an exit if its target label is not somewhere
  // inside the cleanup body region. Gotos whose target label is within the
  // cleanup body stay inside the cleanup scope and need no special handling
  // during flattening; they are simply inlined along with the rest of the
  // body region.
  //
  // This function assigns unique destination IDs to each exit, which are
  // used when multi-exit cleanup scopes are flattened.
  void collectExits(mlir::Region &cleanupBodyRegion,
                    llvm::SmallVectorImpl<CleanupExit> &exits,
                    int &nextId) const {
    // Collect yield terminators from the body region. We do this separately
    // because yields in nested operations, including those in nested cleanup
    // scopes, won't branch through the outer cleanup region.
    for (mlir::Block &block : cleanupBodyRegion) {
      auto *terminator = block.getTerminator();
      if (isa<cir::YieldOp>(terminator))
        exits.emplace_back(terminator, nextId++);
    }

    // Helper to decide whether an op is a goto that needs to be treated as an
    // exit from the cleanup scope being flattened. If op is a goto and targets
    // a label inside the cleanup body region, control stays within the cleanup
    // and we leave the goto in place.
    auto isGotoThatExitsCleanup = [&](mlir::Operation *op) {
      auto gotoOp = dyn_cast<cir::GotoOp>(op);
      return gotoOp && !gotoTargetsLabelInRegion(gotoOp, cleanupBodyRegion);
    };

    // Lambda to walk a loop and collect only returns and gotos.
    // Break and continue inside loops are handled by the loop itself.
    // Loops don't require special handling for nested switch or cleanup scopes
    // because break and continue never branch out of the loop.
    auto collectExitsInLoop = [&](mlir::Operation *loopOp) {
      loopOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *nestedOp) {
        if (isa<cir::ReturnOp>(nestedOp)) {
          exits.emplace_back(nestedOp, nextId++);
        } else if (isGotoThatExitsCleanup(nestedOp)) {
          exits.emplace_back(nestedOp, nextId++);
        }
        return mlir::WalkResult::advance();
      });
    };

    // Forward declaration for mutual recursion.
    std::function<void(mlir::Region &, bool)> collectExitsInCleanup;
    std::function<void(mlir::Operation *)> collectExitsInSwitch;

    // Lambda to collect exits from a switch. Collects return/goto/continue but
    // not break (handled by switch). For nested loops/cleanups, recurses.
    collectExitsInSwitch = [&](mlir::Operation *switchOp) {
      switchOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *nestedOp) {
        if (isa<cir::CleanupScopeOp>(nestedOp)) {
          // Walk the nested cleanup, but ignore break statements because they
          // will be handled by the switch we are currently walking.
          collectExitsInCleanup(
              cast<cir::CleanupScopeOp>(nestedOp).getBodyRegion(),
              /*ignoreBreak=*/true);
          return mlir::WalkResult::skip();
        } else if (isa<cir::LoopOpInterface>(nestedOp)) {
          collectExitsInLoop(nestedOp);
          return mlir::WalkResult::skip();
        } else if (isa<cir::ReturnOp, cir::ContinueOp>(nestedOp)) {
          exits.emplace_back(nestedOp, nextId++);
        } else if (isGotoThatExitsCleanup(nestedOp)) {
          exits.emplace_back(nestedOp, nextId++);
        }
        return mlir::WalkResult::advance();
      });
    };

    // Lambda to collect exits from a cleanup scope body region. This collects
    // break (optionally), continue, return, and goto, handling nested loops,
    // switches, and cleanups appropriately.
    collectExitsInCleanup = [&](mlir::Region &region, bool ignoreBreak) {
      region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
        // We need special handling for break statements because if this cleanup
        // scope was nested within a switch op, break will be handled by the
        // switch operation and therefore won't exit the cleanup scope enclosing
        // the switch. We're only collecting exits from the cleanup that started
        // this walk. Exits from nested cleanups will be handled when we flatten
        // the nested cleanup.
        if (!ignoreBreak && isa<cir::BreakOp>(op)) {
          exits.emplace_back(op, nextId++);
        } else if (isa<cir::ContinueOp, cir::ReturnOp>(op)) {
          exits.emplace_back(op, nextId++);
        } else if (isGotoThatExitsCleanup(op)) {
          exits.emplace_back(op, nextId++);
        } else if (isa<cir::CleanupScopeOp>(op)) {
          // Recurse into nested cleanup's body region.
          collectExitsInCleanup(cast<cir::CleanupScopeOp>(op).getBodyRegion(),
                                /*ignoreBreak=*/ignoreBreak);
          return mlir::WalkResult::skip();
        } else if (isa<cir::LoopOpInterface>(op)) {
          // This kicks off a separate walk rather than continuing to dig deeper
          // in the current walk because we need to handle break and continue
          // differently inside loops.
          collectExitsInLoop(op);
          return mlir::WalkResult::skip();
        } else if (isa<cir::SwitchOp>(op)) {
          // This kicks off a separate walk rather than continuing to dig deeper
          // in the current walk because we need to handle break differently
          // inside switches.
          collectExitsInSwitch(op);
          return mlir::WalkResult::skip();
        }
        return mlir::WalkResult::advance();
      });
    };

    // Collect exits from the body region.
    collectExitsInCleanup(cleanupBodyRegion, /*ignoreBreak=*/false);
  }

  // Check if an operand's defining op should be moved to the destination block.
  // We only sink constants and simple loads. Anything else should be saved
  // to a temporary alloca and reloaded at the destination block.
  static bool shouldSinkReturnOperand(mlir::Value operand,
                                      cir::ReturnOp returnOp) {
    // Block arguments can't be moved
    mlir::Operation *defOp = operand.getDefiningOp();
    if (!defOp)
      return false;

    // Only move constants and loads to the dispatch block. For anything else,
    // we'll store to a temporary and reload in the dispatch block.
    if (!mlir::isa<cir::ConstantOp, cir::LoadOp>(defOp))
      return false;

    // Check if the return is the only user
    if (!operand.hasOneUse())
      return false;

    // Only move ops that are in the same block as the return.
    if (defOp->getBlock() != returnOp->getBlock())
      return false;

    if (auto loadOp = mlir::dyn_cast<cir::LoadOp>(defOp)) {
      // Only attempt to move loads of allocas in the entry block.
      mlir::Value ptr = loadOp.getAddr();
      auto funcOp = returnOp->getParentOfType<cir::FuncOp>();
      assert(funcOp && "Return op has no function parent?");
      mlir::Block &funcEntryBlock = funcOp.getBody().front();

      // Check if it's an alloca in the function entry block
      if (auto allocaOp =
              mlir::dyn_cast_if_present<cir::AllocaOp>(ptr.getDefiningOp()))
        return allocaOp->getBlock() == &funcEntryBlock;

      return false;
    }

    // Make sure we only fall through to here with constants.
    assert(mlir::isa<cir::ConstantOp>(defOp) && "Expected constant op");
    return true;
  }

  // For returns with operands in cleanup dispatch blocks, the operands may not
  // dominate the dispatch block. This function handles that by either sinking
  // the operand's defining op to the dispatch block (for constants and simple
  // loads) or by storing to a temporary alloca and reloading it.
  void
  getReturnOpOperands(cir::ReturnOp returnOp, mlir::Operation *exitOp,
                      mlir::Location loc, mlir::PatternRewriter &rewriter,
                      llvm::SmallVectorImpl<mlir::Value> &returnValues) const {
    mlir::Block *destBlock = rewriter.getInsertionBlock();
    auto funcOp = exitOp->getParentOfType<cir::FuncOp>();
    assert(funcOp && "Return op has no function parent?");
    mlir::Block &funcEntryBlock = funcOp.getBody().front();

    for (mlir::Value operand : returnOp.getOperands()) {
      if (shouldSinkReturnOperand(operand, returnOp)) {
        // Sink the defining op to the dispatch block.
        mlir::Operation *defOp = operand.getDefiningOp();
        rewriter.moveOpBefore(defOp, destBlock, destBlock->end());
        returnValues.push_back(operand);
      } else {
        // Create an alloca in the function entry block.
        cir::AllocaOp alloca;
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&funcEntryBlock);
          cir::CIRDataLayout dataLayout(
              funcOp->getParentOfType<mlir::ModuleOp>());
          uint64_t alignment =
              dataLayout.getAlignment(operand.getType(), true).value();
          cir::PointerType ptrType = cir::PointerType::get(operand.getType());
          alloca = cir::AllocaOp::create(rewriter, loc, ptrType,
                                         operand.getType(), "__ret_operand_tmp",
                                         rewriter.getI64IntegerAttr(alignment));
        }

        // Store the operand value at the original return location.
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(exitOp);
          cir::StoreOp::create(rewriter, loc, operand, alloca,
                               /*isVolatile=*/false,
                               /*alignment=*/mlir::IntegerAttr(),
                               cir::SyncScopeKindAttr(), cir::MemOrderAttr());
        }

        // Reload the value from the temporary alloca in the destination block.
        rewriter.setInsertionPointToEnd(destBlock);
        auto loaded = cir::LoadOp::create(
            rewriter, loc, alloca, /*isDeref=*/false,
            /*isVolatile=*/false, /*alignment=*/mlir::IntegerAttr(),
            cir::SyncScopeKindAttr(), cir::MemOrderAttr());
        returnValues.push_back(loaded);
      }
    }
  }

  // Create the appropriate terminator for an exit operation in the dispatch
  // block. For return ops with operands, this handles the dominance issue by
  // either moving the operand's defining op to the dispatch block (if it's a
  // trivial use) or by storing to a temporary alloca and loading it.
  mlir::LogicalResult
  createExitTerminator(mlir::Operation *exitOp, mlir::Location loc,
                       mlir::Block *continueBlock,
                       mlir::PatternRewriter &rewriter) const {
    return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(exitOp)
        .Case<cir::YieldOp>([&](auto) {
          // Yield becomes a branch to continue block.
          cir::BrOp::create(rewriter, loc, continueBlock);
          return mlir::success();
        })
        .Case<cir::BreakOp>([&](auto) {
          // Break is preserved for later lowering by enclosing switch/loop.
          cir::BreakOp::create(rewriter, loc);
          return mlir::success();
        })
        .Case<cir::ContinueOp>([&](auto) {
          // Continue is preserved for later lowering by enclosing loop.
          cir::ContinueOp::create(rewriter, loc);
          return mlir::success();
        })
        .Case<cir::ReturnOp>([&](auto returnOp) {
          // Return from the cleanup exit. Note, if this is a return inside a
          // nested cleanup scope, the flattening of the outer scope will handle
          // branching through the outer cleanup.
          if (returnOp.hasOperand()) {
            llvm::SmallVector<mlir::Value, 2> returnValues;
            getReturnOpOperands(returnOp, exitOp, loc, rewriter, returnValues);
            cir::ReturnOp::create(rewriter, loc, returnValues);
          } else {
            cir::ReturnOp::create(rewriter, loc);
          }
          return mlir::success();
        })
        .Case<cir::GotoOp>([&](auto gotoOp) {
          // Gotos that target a label within the cleanup body region are
          // filtered out by collectExits and never reach this code, so any
          // goto that does reach here transfers control out of the cleanup
          // scope. The goto is just moved to the exit block.
          cir::GotoOp::create(rewriter, loc, gotoOp.getLabel());
          return mlir::success();
        })
        .Default([&](mlir::Operation *op) {
          cir::UnreachableOp::create(rewriter, loc);
          return op->emitError(
              "unexpected exit operation in cleanup scope body");
        });
  }

#ifndef NDEBUG
  // Check that no block other than the last one in a region exits the region.
  static bool regionExitsOnlyFromLastBlock(mlir::Region &region) {
    for (mlir::Block &block : region) {
      if (&block == &region.back())
        continue;
      bool expectedTerminator =
          llvm::TypeSwitch<mlir::Operation *, bool>(block.getTerminator())
              // It is theoretically possible to have a cleanup block with
              // any of the following exits in non-final blocks, but we won't
              // currently generate any CIR that does that, and being able to
              // assume that it doesn't happen simplifies the implementation.
              // If we ever need to handle this case, the code will need to
              // be updated to handle it.
              .Case<cir::YieldOp, cir::ReturnOp, cir::ResumeFlatOp,
                    cir::ContinueOp, cir::BreakOp, cir::GotoOp>(
                  [](auto) { return false; })
              // We expect that call operations have not yet been rewritten
              // as try_call operations. A call can unwind out of the cleanup
              // scope, but we will be handling that during flattening. The
              // only case where a try_call could be present inside an
              // unflattened cleanup region is if the cleanup contained a
              // nested try-catch region, and that isn't expected as of the
              // time of this implementation. If it does, this could be
              // updated to tolerate it.
              .Case<cir::TryCallOp>([](auto) { return false; })
              // Likewise, we don't expect to find an EH dispatch operation
              // because we weren't expecting try-catch regions nested in the
              // cleanup region.
              .Case<cir::EhDispatchOp>([](auto) { return false; })
              // In theory, it would be possible to have a flattened switch
              // operation that does not exit the cleanup region. For now,
              // that's not happening.
              .Case<cir::SwitchFlatOp>([](auto) { return false; })
              // These aren't expected either, but if they occur, they don't
              // exit the region, so that's OK.
              .Case<cir::UnreachableOp, cir::TrapOp>([](auto) { return true; })
              // Indirect branches are not expected.
              .Case<cir::IndirectBrOp>([](auto) { return false; })
              // We do expect branches, but we don't expect them to leave
              // the region.
              .Case<cir::BrOp>([&](cir::BrOp brOp) {
                assert(brOp.getDest()->getParent() == &region &&
                       "branch destination is not in the region");
                return true;
              })
              .Case<cir::BrCondOp>([&](cir::BrCondOp brCondOp) {
                assert(brCondOp.getDestTrue()->getParent() == &region &&
                       "branch destination is not in the region");
                assert(brCondOp.getDestFalse()->getParent() == &region &&
                       "branch destination is not in the region");
                return true;
              })
              // What else could there be?
              .Default([](mlir::Operation *) -> bool {
                llvm_unreachable("unexpected terminator in cleanup region");
              });
      if (!expectedTerminator)
        return false;
    }
    return true;
  }
#endif

  // Build the EH cleanup block structure by cloning the cleanup region. The
  // cloned entry block gets an !cir.eh_token argument and a cir.begin_cleanup
  // inserted at the top. All cir.yield terminators that might exit the cleanup
  // region are replaced with cir.end_cleanup + cir.resume.
  //
  // For a single-block cleanup region, this produces:
  //
  //   ^eh_cleanup(%eh_token : !cir.eh_token):
  //     %ct = cir.begin_cleanup %eh_token : !cir.eh_token -> !cir.cleanup_token
  //     <cloned cleanup operations>
  //     cir.end_cleanup %ct : !cir.cleanup_token
  //     cir.resume %eh_token : !cir.eh_token
  //
  // For a multi-block cleanup region (e.g. containing a flattened cir.if),
  // the same wrapping is applied around the cloned block structure: the entry
  // block gets begin_cleanup and all exit blocks (those terminated by yield)
  // get end_cleanup + resume.
  //
  // If this cleanup scope is nested within a TryOp, the resume will be updated
  // to branch to the catch dispatch block of the enclosing try operation when
  // the TryOp is flattened.
  mlir::Block *buildEHCleanupBlocks(cir::CleanupScopeOp cleanupOp,
                                    mlir::Location loc,
                                    mlir::Block *insertBefore,
                                    mlir::PatternRewriter &rewriter) const {
    assert(regionExitsOnlyFromLastBlock(cleanupOp.getCleanupRegion()) &&
           "cleanup region has exits in non-final blocks");

    // Track the block before the insertion point so we can find the cloned
    // blocks after cloning.
    mlir::Block *blockBeforeClone = insertBefore->getPrevNode();

    // Clone the entire cleanup region before insertBefore.
    rewriter.cloneRegionBefore(cleanupOp.getCleanupRegion(), insertBefore);

    // Find the first cloned block.
    mlir::Block *clonedEntry = blockBeforeClone
                                   ? blockBeforeClone->getNextNode()
                                   : &insertBefore->getParent()->front();

    // Add the eh_token argument to the cloned entry block and insert
    // begin_cleanup at the top.
    auto ehTokenType = cir::EhTokenType::get(rewriter.getContext());
    mlir::Value ehToken = clonedEntry->addArgument(ehTokenType, loc);

    rewriter.setInsertionPointToStart(clonedEntry);
    auto beginCleanup = cir::BeginCleanupOp::create(rewriter, loc, ehToken);

    // Replace the yield terminator in the last cloned block with
    // end_cleanup + resume.
    mlir::Block *lastClonedBlock = insertBefore->getPrevNode();
    auto yieldOp =
        mlir::dyn_cast<cir::YieldOp>(lastClonedBlock->getTerminator());
    if (yieldOp) {
      rewriter.setInsertionPoint(yieldOp);
      cir::EndCleanupOp::create(rewriter, loc, beginCleanup.getCleanupToken());
      rewriter.replaceOpWithNewOp<cir::ResumeOp>(yieldOp, ehToken);
    } else {
      cleanupOp->emitError("Not yet implemented: cleanup region terminated "
                           "with non-yield operation");
    }

    return clonedEntry;
  }

  // Flatten a cleanup scope. The body region's exits branch to the cleanup
  // block, and the cleanup block branches to destination blocks whose contents
  // depend on the type of operation that exited the body region. Yield becomes
  // a branch to the block after the cleanup scope, break and continue are
  // preserved for later lowering by enclosing switch or loop, and return
  // is preserved as is.
  //
  // If there are multiple exits from the cleanup body, a destination slot and
  // switch dispatch are used to continue to the correct destination after the
  // cleanup is complete. A destination slot alloca is created at the function
  // entry block. Each exit operation is replaced by a store of its unique ID to
  // the destination slot and a branch to cleanup. An operation is appended to
  // the to branch to a dispatch block that loads the destination slot and uses
  // switch.flat to branch to the correct destination.
  //
  // If the cleanup scope requires EH cleanup, any call operations in the body
  // that may throw are replaced with cir.try_call operations that unwind to an
  // EH cleanup block. The cleanup block(s) will be terminated with a cir.resume
  // operation. If this cleanup scope is enclosed by a try operation, the
  // flattening of the try operation flattening will replace the cir.resume with
  // a branch to a catch dispatch block. Otherwise, the cir.resume operation
  // remains in place and will unwind to the caller.
  mlir::LogicalResult
  flattenCleanup(cir::CleanupScopeOp cleanupOp,
                 llvm::SmallVectorImpl<CleanupExit> &exits,
                 llvm::SmallVectorImpl<cir::CallOp> &callsToRewrite,
                 llvm::SmallVectorImpl<cir::ResumeOp> &resumeOpsToChain,
                 mlir::PatternRewriter &rewriter) const {
    mlir::Location loc = cleanupOp.getLoc();
    cir::CleanupKind cleanupKind = cleanupOp.getCleanupKind();
    bool hasNormalCleanup = cleanupKind == cir::CleanupKind::Normal ||
                            cleanupKind == cir::CleanupKind::All;
    bool hasEHCleanup = cleanupKind == cir::CleanupKind::EH ||
                        cleanupKind == cir::CleanupKind::All;
    bool isMultiExit = exits.size() > 1;

    // Get references to region blocks before inlining.
    mlir::Block *bodyEntry = &cleanupOp.getBodyRegion().front();
    mlir::Block *cleanupEntry = &cleanupOp.getCleanupRegion().front();
    mlir::Block *cleanupExit = &cleanupOp.getCleanupRegion().back();
    assert(regionExitsOnlyFromLastBlock(cleanupOp.getCleanupRegion()) &&
           "cleanup region has exits in non-final blocks");
    auto cleanupYield = dyn_cast<cir::YieldOp>(cleanupExit->getTerminator());
    if (!cleanupYield) {
      return rewriter.notifyMatchFailure(cleanupOp,
                                         "Not yet implemented: cleanup region "
                                         "terminated with non-yield operation");
    }

    // For multiple exits from the body region, get or create a destination slot
    // at function entry. The slot is shared across all cleanup scopes in the
    // function. This is only needed if the cleanup scope requires normal
    // cleanup.
    cir::AllocaOp destSlot;
    if (isMultiExit && hasNormalCleanup) {
      auto funcOp = cleanupOp->getParentOfType<cir::FuncOp>();
      if (!funcOp)
        return cleanupOp->emitError("cleanup scope not inside a function");
      destSlot = getOrCreateCleanupDestSlot(funcOp, rewriter, loc);
    }

    // Split the current block to create the insertion point.
    mlir::Block *currentBlock = rewriter.getInsertionBlock();
    mlir::Block *continueBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

    // Build EH cleanup blocks if needed. This must be done before inlining
    // the cleanup region since buildEHCleanupBlocks clones from it. The unwind
    // block is inserted before the EH cleanup entry so that the final layout
    // is: body -> normal cleanup -> exit -> unwind -> EH cleanup -> continue.
    // EH cleanup blocks are needed when there are throwing calls that need to
    // be rewritten to try_call, or when there are resume ops from
    // already-flattened inner cleanup scopes that need to chain through this
    // cleanup's EH handler.
    mlir::Block *unwindBlock = nullptr;
    mlir::Block *ehCleanupEntry = nullptr;
    if (hasEHCleanup &&
        (!callsToRewrite.empty() || !resumeOpsToChain.empty())) {
      ehCleanupEntry =
          buildEHCleanupBlocks(cleanupOp, loc, continueBlock, rewriter);
      // The unwind block is only needed when there are throwing calls that
      // need a shared unwind destination. Resume ops from inner cleanups
      // branch directly to the EH cleanup entry.
      if (!callsToRewrite.empty())
        unwindBlock = buildUnwindBlock(ehCleanupEntry, /*isCleanupOnly=*/true,
                                       loc, ehCleanupEntry, rewriter);
    }

    // All normal flow blocks are inserted before this point — either before
    // the unwind block (if it exists), or before the EH cleanup entry (if EH
    // cleanup exists but no unwind block is needed), or before the continue
    // block.
    mlir::Block *normalInsertPt =
        unwindBlock ? unwindBlock
                    : (ehCleanupEntry ? ehCleanupEntry : continueBlock);

    // Inline the body region.
    rewriter.inlineRegionBefore(cleanupOp.getBodyRegion(), normalInsertPt);

    // Inline the cleanup region for the normal cleanup path.
    if (hasNormalCleanup)
      rewriter.inlineRegionBefore(cleanupOp.getCleanupRegion(), normalInsertPt);

    // Branch from current block to body entry.
    rewriter.setInsertionPointToEnd(currentBlock);
    cir::BrOp::create(rewriter, loc, bodyEntry);

    // Handle normal exits.
    mlir::LogicalResult result = mlir::success();
    if (hasNormalCleanup) {
      // Create the exit/dispatch block (after cleanup, before continue).
      mlir::Block *exitBlock = rewriter.createBlock(normalInsertPt);

      // Rewrite the cleanup region's yield to branch to exit block.
      rewriter.setInsertionPoint(cleanupYield);
      rewriter.replaceOpWithNewOp<cir::BrOp>(cleanupYield, exitBlock);

      if (isMultiExit) {
        // Build the dispatch switch in the exit block.
        rewriter.setInsertionPointToEnd(exitBlock);

        // Load the destination slot value.
        auto slotValue = cir::LoadOp::create(
            rewriter, loc, destSlot, /*isDeref=*/false,
            /*isVolatile=*/false, /*alignment=*/mlir::IntegerAttr(),
            cir::SyncScopeKindAttr(), cir::MemOrderAttr());

        // Create destination blocks for each exit and collect switch case info.
        llvm::SmallVector<mlir::APInt, 8> caseValues;
        llvm::SmallVector<mlir::Block *, 8> caseDestinations;
        llvm::SmallVector<mlir::ValueRange, 8> caseOperands;
        cir::IntType s32Type =
            cir::IntType::get(rewriter.getContext(), 32, /*isSigned=*/true);

        for (const CleanupExit &exit : exits) {
          // Create a block for this destination.
          mlir::Block *destBlock = rewriter.createBlock(normalInsertPt);
          rewriter.setInsertionPointToEnd(destBlock);
          result =
              createExitTerminator(exit.exitOp, loc, continueBlock, rewriter);

          // Add to switch cases.
          caseValues.push_back(
              llvm::APInt(32, static_cast<uint64_t>(exit.destinationId), true));
          caseDestinations.push_back(destBlock);
          caseOperands.push_back(mlir::ValueRange());

          // Replace the original exit op with: store dest ID, branch to
          // cleanup.
          rewriter.setInsertionPoint(exit.exitOp);
          auto destIdConst = cir::ConstantOp::create(
              rewriter, loc, cir::IntAttr::get(s32Type, exit.destinationId));
          cir::StoreOp::create(rewriter, loc, destIdConst, destSlot,
                               /*isVolatile=*/false,
                               /*alignment=*/mlir::IntegerAttr(),
                               cir::SyncScopeKindAttr(), cir::MemOrderAttr());
          rewriter.replaceOpWithNewOp<cir::BrOp>(exit.exitOp, cleanupEntry);

          // If the exit terminator creation failed, we're going to end up with
          // partially flattened code, but we'll also have reported an error so
          // that's OK. We need to finish out this function to keep the IR in a
          // valid state to help diagnose the error. This is a temporary
          // possibility during development. It shouldn't ever happen after the
          // implementation is complete.
          if (result.failed())
            break;
        }

        // Create the default destination (unreachable).
        mlir::Block *defaultBlock = rewriter.createBlock(normalInsertPt);
        rewriter.setInsertionPointToEnd(defaultBlock);
        cir::UnreachableOp::create(rewriter, loc);

        // Build the switch.flat operation in the exit block.
        rewriter.setInsertionPointToEnd(exitBlock);
        cir::SwitchFlatOp::create(rewriter, loc, slotValue, defaultBlock,
                                  mlir::ValueRange(), caseValues,
                                  caseDestinations, caseOperands);
      } else {
        // Single exit: put the appropriate terminator directly in the exit
        // block.
        rewriter.setInsertionPointToEnd(exitBlock);
        mlir::Operation *exitOp = exits[0].exitOp;
        result = createExitTerminator(exitOp, loc, continueBlock, rewriter);

        // Replace body exit with branch to cleanup entry.
        rewriter.setInsertionPoint(exitOp);
        rewriter.replaceOpWithNewOp<cir::BrOp>(exitOp, cleanupEntry);
      }
    } else {
      // EH-only cleanup: normal exits skip the cleanup entirely.
      // Replace yield exits with branches to the continue block.
      for (CleanupExit &exit : exits) {
        if (isa<cir::YieldOp>(exit.exitOp)) {
          rewriter.setInsertionPoint(exit.exitOp);
          rewriter.replaceOpWithNewOp<cir::BrOp>(exit.exitOp, continueBlock);
        }
        // Non-yield exits (break, continue, return) stay as-is since no normal
        // cleanup is needed.
      }
    }

    // Replace non-nothrow calls with try_call operations. All calls within
    // this cleanup scope share the same unwind destination.
    if (hasEHCleanup) {
      for (cir::CallOp callOp : callsToRewrite)
        replaceCallWithTryCall(callOp, unwindBlock, loc, rewriter);
    }

    // Handle throwing calls in EH cleanup blocks. When an exception is thrown
    // during cleanup code that runs on the exception unwind path, the C++
    // standard requires that std::terminate() be called. Replace such calls
    // with try_call operations that unwind to a terminate block containing
    // cir.eh.initiate + cir.eh.terminate.
    if (ehCleanupEntry) {
      llvm::SmallVector<cir::CallOp> ehCleanupThrowingCalls;
      for (mlir::Block *block = ehCleanupEntry; block != continueBlock;
           block = block->getNextNode()) {
        block->walk([&](cir::CallOp callOp) {
          if (!callOp.getNothrow())
            ehCleanupThrowingCalls.push_back(callOp);
        });
      }
      if (!ehCleanupThrowingCalls.empty()) {
        mlir::Block *terminateBlock =
            buildTerminateUnwindBlock(loc, continueBlock, rewriter);
        for (cir::CallOp callOp : ehCleanupThrowingCalls)
          replaceCallWithTryCall(callOp, terminateBlock, loc, rewriter);
      }
    }

    // Chain inner EH cleanup resume ops to this cleanup's EH handler.
    // Each cir.resume from an already-flattened inner cleanup is replaced
    // with a branch to the outer EH cleanup entry, passing the eh_token
    // from the inner's begin_cleanup so that the same in-flight exception
    // flows through the outer cleanup before unwinding to the caller.
    if (ehCleanupEntry) {
      for (cir::ResumeOp resumeOp : resumeOpsToChain) {
        mlir::Value ehToken = resumeOp.getEhToken();
        rewriter.setInsertionPoint(resumeOp);
        rewriter.replaceOpWithNewOp<cir::BrOp>(
            resumeOp, mlir::ValueRange{ehToken}, ehCleanupEntry);
      }
    }

    // Erase the original cleanup scope op.
    rewriter.eraseOp(cleanupOp);

    // Always return success because the IR has been modified (blocks split,
    // regions inlined, ops erased, etc.). The MLIR pattern rewriter contract
    // requires that if a pattern modifies IR, it must return success().
    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewrite(cir::CleanupScopeOp cleanupOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    // All nested structured CIR ops must be flattened before the cleanup scope.
    // Operations like loops, switches, scopes, and ifs may contain exits
    // (return, break, continue) that the cleanup scope will replace with
    // branches to the cleanup entry. If those exits are inside a structured
    // op's region, the branch would reference a block outside that region,
    // which is invalid. Fail the match so they are processed first.
    //
    // Before checking, erase any trivially dead nested cleanup scopes. These
    // arise from deactivated cleanups (e.g. partial-construction guards for
    // lambda captures). The greedy rewriter may have already DCE'd them, but
    // when a trivially dead nested op is erased first, the parent isn't always
    // re-added to the worklist, so we handle it here.
    llvm::SmallVector<cir::CleanupScopeOp> deadNestedOps;
    cleanupOp.getBodyRegion().walk([&](cir::CleanupScopeOp nested) {
      if (mlir::isOpTriviallyDead(nested))
        deadNestedOps.push_back(nested);
    });
    for (auto op : deadNestedOps)
      rewriter.eraseOp(op);

    if (hasNestedOpsToFlatten(cleanupOp.getBodyRegion()))
      return mlir::failure();

    cir::CleanupKind cleanupKind = cleanupOp.getCleanupKind();

    // Collect all exits from the body region.
    llvm::SmallVector<CleanupExit> exits;
    int nextId = 0;
    collectExits(cleanupOp.getBodyRegion(), exits, nextId);

    assert(!exits.empty() && "cleanup scope body has no exit");

    // Collect non-nothrow calls that need to be converted to try_call.
    // This is only needed for EH and All cleanup kinds, but the vector
    // will simply be empty for Normal cleanup.
    llvm::SmallVector<cir::CallOp> callsToRewrite;
    if (cleanupKind != cir::CleanupKind::Normal)
      collectThrowingCalls(cleanupOp.getBodyRegion(), callsToRewrite);

    // Collect resume ops from already-flattened inner cleanup scopes that
    // need to chain through this cleanup's EH handler.
    llvm::SmallVector<cir::ResumeOp> resumeOpsToChain;
    if (cleanupKind != cir::CleanupKind::Normal)
      collectResumeOps(cleanupOp.getBodyRegion(), resumeOpsToChain);

    return flattenCleanup(cleanupOp, exits, callsToRewrite, resumeOpsToChain,
                          rewriter);
  }
};

// Trace an !cir.eh_token value back through block arguments to find the
// cir.eh.initiate operation that defines it. Returns {} if the defining op
// cannot be found (e.g. multiple predecessors).
static cir::EhInitiateOp traceToEhInitiate(mlir::Value ehToken) {
  while (ehToken) {
    if (auto initiate = ehToken.getDefiningOp<cir::EhInitiateOp>())
      return initiate;
    auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(ehToken);
    if (!blockArg)
      return {};
    mlir::Block *pred = blockArg.getOwner()->getSinglePredecessor();
    if (!pred)
      return {};
    auto brOp = mlir::dyn_cast<cir::BrOp>(pred->getTerminator());
    if (!brOp)
      return {};
    ehToken = brOp.getDestOperands()[blockArg.getArgNumber()];
  }
  return {};
}

class CIRTryOpFlattening : public mlir::OpRewritePattern<cir::TryOp> {
public:
  using OpRewritePattern<cir::TryOp>::OpRewritePattern;

  // Build the catch dispatch block with a cir.eh.dispatch operation.
  // The dispatch block receives an !cir.eh_token argument and dispatches
  // to the appropriate catch handler blocks based on exception types.
  mlir::Block *buildCatchDispatchBlock(
      cir::TryOp tryOp, mlir::ArrayAttr handlerTypes,
      llvm::SmallVectorImpl<mlir::Block *> &catchHandlerBlocks,
      mlir::Location loc, mlir::Block *insertBefore,
      mlir::PatternRewriter &rewriter) const {
    mlir::Block *dispatchBlock = rewriter.createBlock(insertBefore);
    auto ehTokenType = cir::EhTokenType::get(rewriter.getContext());
    mlir::Value ehToken = dispatchBlock->addArgument(ehTokenType, loc);

    rewriter.setInsertionPointToEnd(dispatchBlock);

    // Build the catch types and destinations for the dispatch.
    llvm::SmallVector<mlir::Attribute> catchTypeAttrs;
    llvm::SmallVector<mlir::Block *> catchDests;
    mlir::Block *defaultDest = nullptr;
    bool defaultIsCatchAll = false;

    for (auto [typeAttr, handlerBlock] :
         llvm::zip(handlerTypes, catchHandlerBlocks)) {
      if (mlir::isa<cir::CatchAllAttr>(typeAttr)) {
        assert(!defaultDest && "multiple catch_all or unwind handlers");
        defaultDest = handlerBlock;
        defaultIsCatchAll = true;
      } else if (mlir::isa<cir::UnwindAttr>(typeAttr)) {
        assert(!defaultDest && "multiple catch_all or unwind handlers");
        defaultDest = handlerBlock;
        defaultIsCatchAll = false;
      } else {
        // This is a typed catch handler (GlobalViewAttr with type info).
        catchTypeAttrs.push_back(typeAttr);
        catchDests.push_back(handlerBlock);
      }
    }

    assert(defaultDest && "dispatch must have a catch_all or unwind handler");

    mlir::ArrayAttr catchTypesArrayAttr;
    if (!catchTypeAttrs.empty())
      catchTypesArrayAttr = rewriter.getArrayAttr(catchTypeAttrs);

    cir::EhDispatchOp::create(rewriter, loc, ehToken, catchTypesArrayAttr,
                              defaultIsCatchAll, defaultDest, catchDests);

    return dispatchBlock;
  }

  // Flatten a single catch handler region. Each handler region has an
  // !cir.eh_token argument and starts with cir.begin_catch, followed by
  // a cir.cleanup.scope containing the handler body (with cir.end_catch in
  // its cleanup region), and ending with cir.yield.
  //
  // After flattening, the handler region becomes a block that receives the
  // eh_token, calls begin_catch, runs the handler body inline, calls
  // end_catch, and branches to the continue block.
  //
  // The cleanup scope inside the catch handler is expected to have been
  // flattened before we get here, so what we see in the handler region is
  // already flat code with begin_catch at the top and end_catch in any place
  // that we would exit the catch handler. We just need to inline the region
  // and fix up terminators.
  mlir::Block *flattenCatchHandler(mlir::Region &handlerRegion,
                                   mlir::Block *continueBlock,
                                   mlir::Location loc,
                                   mlir::Block *insertBefore,
                                   mlir::PatternRewriter &rewriter) const {
    // The handler region entry block has the !cir.eh_token argument.
    mlir::Block *handlerEntry = &handlerRegion.front();

    // Inline the handler region before insertBefore.
    rewriter.inlineRegionBefore(handlerRegion, insertBefore);

    // Replace yield terminators in the handler with branches to continue.
    for (mlir::Block &block : llvm::make_range(handlerEntry->getIterator(),
                                               insertBefore->getIterator())) {
      if (auto yieldOp = dyn_cast<cir::YieldOp>(block.getTerminator())) {
        // Verify that end_catch is the last non-branch operation before
        // this yield.  After cleanup scope flattening, end_catch may be
        // in a predecessor block rather than immediately before the yield.
        // Walk back through predecessors (including multi-predecessor
        // blocks), verifying that each intermediate block contains only a
        // branch terminator, until we find end_catch as the last
        // non-terminator in some block.
        // Verify that end_catch is reachable on some predecessor path
        // before this yield.  After cleanup scope flattening, end_catch
        // may be separated from yield by conditional branches (e.g.,
        // from flattened cir.if inside the catch body).
        assert(([&]() {
                 if (mlir::Operation *prev = yieldOp->getPrevNode())
                   return isa<cir::EndCatchOp>(prev);
                 llvm::SmallPtrSet<mlir::Block *, 8> visited;
                 llvm::SmallVector<mlir::Block *, 4> worklist;
                 for (mlir::Block *pred : block.getPredecessors())
                   worklist.push_back(pred);
                 while (!worklist.empty()) {
                   mlir::Block *b = worklist.pop_back_val();
                   if (!visited.insert(b).second)
                     continue;
                   mlir::Operation *term = b->getTerminator();
                   if (mlir::Operation *prev = term->getPrevNode()) {
                     if (isa<cir::EndCatchOp>(prev))
                       return true;
                   }
                   for (mlir::Block *pred : b->getPredecessors())
                     worklist.push_back(pred);
                 }
                 return false;
               }()) &&
               "expected end_catch reachable before yield "
               "in catch handler");
        rewriter.setInsertionPoint(yieldOp);
        rewriter.replaceOpWithNewOp<cir::BrOp>(yieldOp, continueBlock);
      }
    }

    return handlerEntry;
  }

  // Flatten an unwind handler region. The unwind region just contains a
  // cir.resume that continues unwinding. We inline it and leave the resume
  // in place. If this try op is nested inside an EH cleanup or another try op,
  // the enclosing op will rewrite the resume as a branch to its cleanup or
  // dispatch block when it is flattened. Otherwise, the resume will unwind to
  // the caller.
  mlir::Block *flattenUnwindHandler(mlir::Region &unwindRegion,
                                    mlir::Location loc,
                                    mlir::Block *insertBefore,
                                    mlir::PatternRewriter &rewriter) const {
    mlir::Block *unwindEntry = &unwindRegion.front();
    rewriter.inlineRegionBefore(unwindRegion, insertBefore);
    return unwindEntry;
  }

  mlir::LogicalResult
  matchAndRewrite(cir::TryOp tryOp,
                  mlir::PatternRewriter &rewriter) const override {
    // All nested structured CIR ops must be flattened before the try op.
    // Cleanup scopes and nested try ops need to be flat so EH cleanup is
    // properly handled. Other structured ops (scopes, ifs, loops, switches,
    // ternaries) must be flat because replaceCallWithTryCall creates try_call
    // ops whose unwind destination is outside the structured op's region,
    // which would be an invalid cross-region reference.
    for (mlir::Region &region : tryOp->getRegions())
      if (hasNestedOpsToFlatten(region))
        return mlir::failure();

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    mlir::Location loc = tryOp.getLoc();

    mlir::ArrayAttr handlerTypes = tryOp.getHandlerTypesAttr();
    mlir::MutableArrayRef<mlir::Region> handlerRegions =
        tryOp.getHandlerRegions();

    // Collect throwing calls in the try body.
    llvm::SmallVector<cir::CallOp> callsToRewrite;
    collectThrowingCalls(tryOp.getTryRegion(), callsToRewrite);

    // Collect resume ops from already-flattened cleanup scopes in the try body.
    llvm::SmallVector<cir::ResumeOp> resumeOpsToChain;
    collectResumeOps(tryOp.getTryRegion(), resumeOpsToChain);

    // Split the current block and inline the try body.
    mlir::Block *currentBlock = rewriter.getInsertionBlock();
    mlir::Block *continueBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

    // Get references to try body blocks before inlining.
    mlir::Block *bodyEntry = &tryOp.getTryRegion().front();
    mlir::Block *bodyExit = &tryOp.getTryRegion().back();

    // Inline the try body region before the continue block.
    rewriter.inlineRegionBefore(tryOp.getTryRegion(), continueBlock);

    // Branch from the current block to the body entry.
    rewriter.setInsertionPointToEnd(currentBlock);
    cir::BrOp::create(rewriter, loc, bodyEntry);

    // Replace the try body's yield terminator with a branch to continue.
    if (auto bodyYield = dyn_cast<cir::YieldOp>(bodyExit->getTerminator())) {
      rewriter.setInsertionPoint(bodyYield);
      rewriter.replaceOpWithNewOp<cir::BrOp>(bodyYield, continueBlock);
    }

    // If there are no handlers, we're done.
    if (!handlerTypes || handlerTypes.empty()) {
      rewriter.eraseOp(tryOp);
      return mlir::success();
    }

    // If there are no throwing calls and no resume ops from inner cleanup
    // scopes, exceptions cannot reach the catch handlers. Drop all uses
    // from the (unreachable) handler regions before erasing the try op,
    // since handler ops may reference values that were inlined from the
    // try body into the parent block.
    if (callsToRewrite.empty() && resumeOpsToChain.empty()) {
      for (mlir::Region &handlerRegion : handlerRegions)
        for (mlir::Block &block : handlerRegion)
          block.dropAllDefinedValueUses();
      rewriter.eraseOp(tryOp);
      return mlir::success();
    }

    // Build the catch handler blocks.

    // First, flatten all handler regions and collect the entry blocks.
    llvm::SmallVector<mlir::Block *> catchHandlerBlocks;

    for (const auto &[idx, typeAttr] : llvm::enumerate(handlerTypes)) {
      mlir::Region &handlerRegion = handlerRegions[idx];

      if (mlir::isa<cir::UnwindAttr>(typeAttr)) {
        mlir::Block *unwindEntry =
            flattenUnwindHandler(handlerRegion, loc, continueBlock, rewriter);
        catchHandlerBlocks.push_back(unwindEntry);
      } else {
        mlir::Block *handlerEntry = flattenCatchHandler(
            handlerRegion, continueBlock, loc, continueBlock, rewriter);
        catchHandlerBlocks.push_back(handlerEntry);
      }
    }

    // Build the catch dispatch block.
    mlir::Block *dispatchBlock =
        buildCatchDispatchBlock(tryOp, handlerTypes, catchHandlerBlocks, loc,
                                catchHandlerBlocks.front(), rewriter);

    // Check whether the try has a catch-all handler. When catch-all is
    // present, the personality function will always stop unwinding at this
    // frame (because catch-all matches every exception type). The LLVM
    // landingpad therefore needs "catch ptr null" rather than "cleanup".
    // The downstream pipeline (EHABILowering + LowerToLLVM) emits
    // "catch ptr null" when the EhInitiateOp has neither cleanup nor typed
    // catch types, so we clear the cleanup flag on every EhInitiateOp that
    // feeds into a dispatch with a catch-all handler.
    bool hasCatchAll =
        handlerTypes && llvm::any_of(handlerTypes, [](mlir::Attribute attr) {
          return mlir::isa<cir::CatchAllAttr>(attr);
        });

    // Build a block to be the unwind desination for throwing calls and replace
    // the calls with try_call ops. Note that the unwind block created here is
    // something different than the unwind handler that we may have created
    // above. The unwind handler continues unwinding after uncaught exceptions.
    // This is the block that will eventually become the landing pad for invoke
    // instructions.
    bool isCleanupOnly = tryOp.getCleanup() && !hasCatchAll;
    if (!callsToRewrite.empty()) {
      // Create a shared unwind block for all throwing calls.
      mlir::Block *unwindBlock = buildUnwindBlock(dispatchBlock, isCleanupOnly,
                                                  loc, dispatchBlock, rewriter);

      for (cir::CallOp callOp : callsToRewrite)
        replaceCallWithTryCall(callOp, unwindBlock, loc, rewriter);
    }

    // Chain resume ops from inner cleanup scopes.
    // Resume ops from already-flattened cleanup scopes within the try body
    // should branch to the catch dispatch block instead of unwinding directly.
    for (cir::ResumeOp resumeOp : resumeOpsToChain) {
      // When there is a catch-all handler, clear the cleanup flag on the
      // cir.eh.initiate that produced this token. With catch-all, the LLVM
      // landingpad needs "catch ptr null" instead of "cleanup".
      if (hasCatchAll) {
        if (auto ehInitiate = traceToEhInitiate(resumeOp.getEhToken())) {
          rewriter.modifyOpInPlace(ehInitiate,
                                   [&] { ehInitiate.removeCleanupAttr(); });
        }
      }

      mlir::Value ehToken = resumeOp.getEhToken();
      rewriter.setInsertionPoint(resumeOp);
      rewriter.replaceOpWithNewOp<cir::BrOp>(
          resumeOp, mlir::ValueRange{ehToken}, dispatchBlock);
    }

    // Finally, erase the original try op ----
    rewriter.eraseOp(tryOp);

    return mlir::success();
  }
};

void populateFlattenCFGPatterns(RewritePatternSet &patterns) {
  patterns
      .add<CIRIfFlattening, CIRLoopOpInterfaceFlattening, CIRScopeOpFlattening,
           CIRSwitchOpFlattening, CIRTernaryOpFlattening,
           CIRCleanupScopeOpFlattening, CIRTryOpFlattening>(
          patterns.getContext());
}

void CIRFlattenCFGPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateFlattenCFGPatterns(patterns);

  // Collect operations to apply patterns.
  llvm::SmallVector<Operation *, 16> ops;
  getOperation()->walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
    if (isa<IfOp, ScopeOp, SwitchOp, LoopOpInterface, TernaryOp, CleanupScopeOp,
            TryOp>(op))
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

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

// TODO(cir): Move CleanupExit and collectExits into
//    CIRCleanupScopeOpFlattening after multi-exit handling is implemented.
//    They're here for now so that we can use them to emit errors for the
//    not-yet-implemented multi-exit case.

struct CleanupExit {
  // An operation that exits the cleanup scope (yield, break, continue,
  // return, etc.)
  mlir::Operation *exitOp;

  // A unique identifier for this exit's destination (used for switch dispatch
  // when there are multiple exits).
  int destinationId;

  CleanupExit(mlir::Operation *op, int id) : exitOp(op), destinationId(id) {}
};

// Collect all operations that exit a cleanup scope body. Return, goto, break,
// and continue can all require branches through the cleanup region. When a loop
// is encountered, only return and goto are collected because break and continue
// are handled by the loop and stay within the cleanup scope. When a switch is
// encountered, return, goto and continue are collected because they may all
// branch through the cleanup, but break is local to the switch. When a nested
// cleanup scope is encountered, we recursively collect exits since any return,
// goto, break, or continue from the nested cleanup will also branch through the
// outer cleanup.
//
// Note that goto statements may not necessarily exit the cleanup scope, but
// for now we conservatively assume that they do. We'll need more nuanced
// handling of that when multi-exit flattening is implemented.
//
// This function assigns unique destination IDs to each exit, which will be used
// when multi-exit flattening is implemented.
static void collectExits(mlir::Region &cleanupBodyRegion,
                         llvm::SmallVectorImpl<CleanupExit> &exits,
                         int &nextId) {
  // Collect yield terminators from the body region. We do this separately
  // because yields in nested operations, including those in nested cleanup
  // scopes, won't branch through the outer cleanup region.
  for (mlir::Block &block : cleanupBodyRegion) {
    auto *terminator = block.getTerminator();
    if (isa<cir::YieldOp>(terminator))
      exits.emplace_back(terminator, nextId++);
  }

  // Lambda to walk a loop and collect only returns and gotos.
  // Break and continue inside loops are handled by the loop itself.
  // Loops don't require special handling for nested switch or cleanup scopes
  // because break and continue never branch out of the loop.
  auto collectExitsInLoop = [&](mlir::Operation *loopOp) {
    loopOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *nestedOp) {
      if (isa<cir::ReturnOp, cir::GotoOp>(nestedOp))
        exits.emplace_back(nestedOp, nextId++);
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
      } else if (isa<cir::ReturnOp, cir::GotoOp, cir::ContinueOp>(nestedOp)) {
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
      } else if (isa<cir::ContinueOp, cir::ReturnOp, cir::GotoOp>(op)) {
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

// Check if this operation is within a cleanup scope or contains a cleanup
// scope with multiple exits. Either of these are unimplemented conditions and
// should trigger an error for now. This is a temporary check that is only
// needed until multi-exit cleanup flattening is implemented.
static bool enclosedByCleanupScopeWithMultipleExits(mlir::Operation *op) {
  int nextId = 0;
  cir::CleanupScopeOp cleanupParent =
      op->getParentOfType<cir::CleanupScopeOp>();
  if (!cleanupParent)
    return false;
  llvm::SmallVector<CleanupExit> exits;
  collectExits(cleanupParent.getBodyRegion(), exits, nextId);
  if (exits.size() > 1)
    return true;
  return false;
}

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
    cir::BinOp diffValue =
        cir::BinOp::create(rewriter, op.getLoc(), sIntType, cir::BinOpKind::Sub,
                           op.getCondition(), lowerBoundValue);

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
    // Cleanup scopes must be lowered before the enclosing switch so that
    // break inside them is properly routed through cleanup.
    // Fail the match so the pattern rewriter will process cleanup scopes first.
    bool hasNestedCleanup = false;
    op->walk([&](cir::CleanupScopeOp) { hasNestedCleanup = true; });
    if (hasNestedCleanup)
      return mlir::failure();

    // Don't flatten switches that contain cleanup scopes with multiple exits
    // (break/continue/return/goto). Those cleanup scopes need multi-exit
    // handling (destination slot + switch dispatch) which is not yet
    // implemented.
    if (enclosedByCleanupScopeWithMultipleExits(op))
      return op->emitError("Cannot lower switch: cleanup with multiple exits");

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
    // Cleanup scopes must be lowered before the enclosing loop so that
    // break/continue inside them are properly routed through cleanup.
    // Fail the match so the pattern rewriter will process cleanup scopes first.
    bool hasNestedCleanup = false;
    op->walk([&](cir::CleanupScopeOp) { hasNestedCleanup = true; });
    if (hasNestedCleanup)
      return mlir::failure();

    // Don't flatten loops that contain cleanup scopes with multiple exits
    // (break/continue/return/goto). Those cleanup scopes need multi-exit
    // handling (destination slot + switch dispatch) which is not yet
    // implemented.
    if (enclosedByCleanupScopeWithMultipleExits(op))
      return op->emitError("Cannot lower loop: cleanup with multiple exits");

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
    cir::BrOp::create(rewriter, loc, remainingOpsBlock);

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
    cir::BrCondOp::create(rewriter, loc, op.getCond(), trueBlock, falseBlock);

    rewriter.replaceOp(op, continueBlock->getArguments());

    // Ok, we're done!
    return mlir::success();
  }
};

class CIRCleanupScopeOpFlattening
    : public mlir::OpRewritePattern<cir::CleanupScopeOp> {
public:
  using OpRewritePattern<cir::CleanupScopeOp>::OpRewritePattern;

  // Flatten a cleanup scope with a single exit destination.
  // The body region's exit branches to the cleanup block, the cleanup block
  // branches to a cleanup exit block whose contents depend on the type of
  // operation that exited the body region. Yield becomes a branch to the
  // block after the cleanup scope, break and continue are preserved
  // for later lowering by enclosing switch or loop. Return is preserved as is.
  mlir::LogicalResult
  flattenSimpleCleanup(cir::CleanupScopeOp cleanupOp, mlir::Operation *exitOp,
                       mlir::PatternRewriter &rewriter) const {
    mlir::Location loc = cleanupOp.getLoc();

    // Get references to region blocks before inlining.
    mlir::Block *bodyEntry = &cleanupOp.getBodyRegion().front();
    mlir::Block *cleanupEntry = &cleanupOp.getCleanupRegion().front();
    mlir::Block *cleanupExit = &cleanupOp.getCleanupRegion().back();

    auto cleanupYield = dyn_cast<cir::YieldOp>(cleanupExit->getTerminator());
    if (!cleanupYield) {
      return rewriter.notifyMatchFailure(cleanupOp,
                                         "Not yet implemented: cleanup region "
                                         "terminated with non-yield operation");
    }

    // Split the current block to create the insertion point.
    mlir::Block *currentBlock = rewriter.getInsertionBlock();
    mlir::Block *continueBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

    // Inline the body region.
    rewriter.inlineRegionBefore(cleanupOp.getBodyRegion(), continueBlock);

    // Inline the cleanup region after the body.
    rewriter.inlineRegionBefore(cleanupOp.getCleanupRegion(), continueBlock);

    // Branch from current block to body entry.
    rewriter.setInsertionPointToEnd(currentBlock);
    cir::BrOp::create(rewriter, loc, bodyEntry);

    // Create a block for the exit terminator (after cleanup, before continue).
    mlir::Block *exitBlock = rewriter.createBlock(continueBlock);

    // Rewrite the cleanup region's yield to branch to exit block.
    rewriter.setInsertionPoint(cleanupYield);
    rewriter.replaceOpWithNewOp<cir::BrOp>(cleanupYield, exitBlock);

    // Put the appropriate terminator in the exit block.
    rewriter.setInsertionPointToEnd(exitBlock);
    llvm::TypeSwitch<mlir::Operation *, void>(exitOp)
        .Case<cir::YieldOp>([&](auto) {
          // Yield becomes a branch to continue block.
          cir::BrOp::create(rewriter, loc, continueBlock);
        })
        .Case<cir::BreakOp>([&](auto) {
          // Break is preserved for later lowering by enclosing switch/loop.
          cir::BreakOp::create(rewriter, loc);
        })
        .Case<cir::ContinueOp>([&](auto) {
          // Continue is preserved for later lowering by enclosing loop.
          cir::ContinueOp::create(rewriter, loc);
        })
        .Case<cir::ReturnOp>([&](auto returnOp) {
          // Return from the cleanup exit. Note, if this is a return inside a
          // nested cleanup scope, the flattening of the outer scope will handle
          // branching through the outer cleanup.
          if (returnOp.hasOperand())
            cir::ReturnOp::create(rewriter, loc, returnOp.getOperands());
          else
            cir::ReturnOp::create(rewriter, loc);
        })
        .Default([&](mlir::Operation *op) {
          op->emitError("unexpected terminator in cleanup scope body");
        });

    // Replace body exit with branch to cleanup entry.
    rewriter.setInsertionPoint(exitOp);
    rewriter.replaceOpWithNewOp<cir::BrOp>(exitOp, cleanupEntry);

    // Erase the original cleanup scope op.
    rewriter.eraseOp(cleanupOp);

    return mlir::success();
  }

  // Flatten a cleanup scope with multiple exit destinations.
  // Uses a destination slot and switch dispatch after cleanup.
  mlir::LogicalResult
  flattenMultiExitCleanup(cir::CleanupScopeOp cleanupOp,
                          llvm::SmallVectorImpl<CleanupExit> &exits,
                          mlir::PatternRewriter &rewriter) const {
    // This will implement the destination slot mechanism:
    // 1. Allocate a destination slot at function entry
    // 2. Each exit stores its destination ID to the slot
    // 3. All exits branch to cleanup entry
    // 4. Cleanup branches to a dispatch block
    // 5. Dispatch block loads slot and switches to correct destination
    //
    // For now, we report this as a match failure and leave the cleanup scope
    // unchanged. The cleanup scope must remain inside its enclosing loop so
    // that break/continue ops remain valid.
    return cleanupOp->emitError(
        "cleanup scope with multiple exits is not yet implemented");
  }

  mlir::LogicalResult
  matchAndRewrite(cir::CleanupScopeOp cleanupOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    // Only handle normal cleanups for now - EH and "all" cleanups are NYI.
    cir::CleanupKind cleanupKind = cleanupOp.getCleanupKind();
    if (cleanupKind != cir::CleanupKind::Normal)
      return cleanupOp->emitError(
          "EH cleanup flattening is not yet implemented");

    // Collect all exits from the body region.
    llvm::SmallVector<CleanupExit> exits;
    int nextId = 0;
    collectExits(cleanupOp.getBodyRegion(), exits, nextId);

    if (exits.size() > 1)
      return flattenMultiExitCleanup(cleanupOp, exits, rewriter);

    assert(!exits.empty() && "cleanup scope body has no exit");

    return flattenSimpleCleanup(cleanupOp, exits[0].exitOp, rewriter);
  }
};

class CIRTryOpFlattening : public mlir::OpRewritePattern<cir::TryOp> {
public:
  using OpRewritePattern<cir::TryOp>::OpRewritePattern;

  mlir::Block *buildTryBody(cir::TryOp tryOp,
                            mlir::PatternRewriter &rewriter) const {
    // Split the current block before the TryOp to create the inlining
    // point.
    mlir::Block *beforeTryScopeBlock = rewriter.getInsertionBlock();
    mlir::Block *afterTry =
        rewriter.splitBlock(beforeTryScopeBlock, rewriter.getInsertionPoint());

    // Inline body region.
    mlir::Block *beforeBody = &tryOp.getTryRegion().front();
    rewriter.inlineRegionBefore(tryOp.getTryRegion(), afterTry);

    // Branch into the body of the region.
    rewriter.setInsertionPointToEnd(beforeTryScopeBlock);
    cir::BrOp::create(rewriter, tryOp.getLoc(), mlir::ValueRange(), beforeBody);
    return afterTry;
  }

  void buildHandlers(cir::TryOp tryOp, mlir::PatternRewriter &rewriter,
                     mlir::Block *afterBody, mlir::Block *afterTry,
                     SmallVectorImpl<cir::CallOp> &callsToRewrite,
                     SmallVectorImpl<mlir::Block *> &landingPads) const {
    // Replace the tryOp return with a branch that jumps out of the body.
    rewriter.setInsertionPointToEnd(afterBody);

    mlir::Block *beforeCatch = rewriter.getInsertionBlock();
    rewriter.setInsertionPointToEnd(beforeCatch);

    // Check if the terminator is a YieldOp because there could be another
    // terminator, e.g. unreachable
    if (auto tryBodyYield = dyn_cast<cir::YieldOp>(afterBody->getTerminator()))
      rewriter.replaceOpWithNewOp<cir::BrOp>(tryBodyYield, afterTry);

    mlir::ArrayAttr handlers = tryOp.getHandlerTypesAttr();
    if (!handlers || handlers.empty())
      return;

    llvm_unreachable("TryOpFlattening buildHandlers with CallsOp is NYI");
  }

  mlir::LogicalResult
  matchAndRewrite(cir::TryOp tryOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    mlir::Block *afterBody = &tryOp.getTryRegion().back();

    // Grab the collection of `cir.call exception`s to rewrite to
    // `cir.try_call`.
    llvm::SmallVector<cir::CallOp, 4> callsToRewrite;
    tryOp.getTryRegion().walk([&](CallOp op) {
      if (op.getNothrow())
        return;

      // Only grab calls within immediate closest TryOp scope.
      if (op->getParentOfType<cir::TryOp>() != tryOp)
        return;
      callsToRewrite.push_back(op);
    });

    if (!callsToRewrite.empty())
      llvm_unreachable(
          "TryOpFlattening with try block that contains CallOps is NYI");

    // Build try body.
    mlir::Block *afterTry = buildTryBody(tryOp, rewriter);

    // Build handlers.
    llvm::SmallVector<mlir::Block *, 4> landingPads;
    buildHandlers(tryOp, rewriter, afterBody, afterTry, callsToRewrite,
                  landingPads);

    rewriter.eraseOp(tryOp);

    assert((landingPads.size() == callsToRewrite.size()) &&
           "expected matching number of entries");

    // Quick block cleanup: no indirection to the post try block.
    auto brOp = dyn_cast<cir::BrOp>(afterTry->getTerminator());
    if (brOp && brOp.getDest()->hasNoPredecessors()) {
      mlir::Block *srcBlock = brOp.getDest();
      rewriter.eraseOp(brOp);
      rewriter.mergeBlocks(srcBlock, afterTry);
    }

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

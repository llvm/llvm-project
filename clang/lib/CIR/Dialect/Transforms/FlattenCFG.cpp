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
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
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
    bool hasNestedCleanup = op->walk([&](cir::CleanupScopeOp) {
                                return mlir::WalkResult::interrupt();
                              }).wasInterrupted();
    if (hasNestedCleanup)
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
    // Cleanup scopes must be lowered before the enclosing loop so that
    // break/continue inside them are properly routed through cleanup.
    // Fail the match so the pattern rewriter will process cleanup scopes first.
    bool hasNestedCleanup = op->walk([&](cir::CleanupScopeOp) {
                                return mlir::WalkResult::interrupt();
                              }).wasInterrupted();
    if (hasNestedCleanup)
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
  // Note that goto statements may not necessarily exit the cleanup scope, but
  // for now we conservatively assume that they do. We'll need more nuanced
  // handling of that when multi-exit flattening is implemented.
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
        defOp->moveBefore(destBlock, destBlock->end());
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
          // Correct goto handling requires determining whether the goto
          // branches out of the cleanup scope or stays within it.
          // Although the goto necessarily exits the cleanup scope in the
          // case where it is the only exit from the scope, it is left
          // as unimplemented for now so that it can be generalized when
          // multi-exit flattening is implemented.
          cir::UnreachableOp::create(rewriter, loc);
          return gotoOp.emitError(
              "goto in cleanup scope is not yet implemented");
        })
        .Default([&](mlir::Operation *op) {
          cir::UnreachableOp::create(rewriter, loc);
          return op->emitError(
              "unexpected exit operation in cleanup scope body");
        });
  }

  // Collect all function calls in the cleanup scope body that may throw
  // exceptions and need to be replaced with try_call operations. Skips calls
  // that are marked nothrow and calls inside nested TryOps (the latter will be
  // handled by the TryOp's own flattening).
  void collectThrowingCalls(
      mlir::Region &bodyRegion,
      llvm::SmallVectorImpl<cir::CallOp> &callsToRewrite) const {
    bodyRegion.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
      // Skip calls inside nested TryOps - those are handled by TryOp
      // flattening.
      if (isa<cir::TryOp>(op))
        return mlir::WalkResult::skip();

      if (auto callOp = dyn_cast<cir::CallOp>(op)) {
        if (!callOp.getNothrow())
          callsToRewrite.push_back(callOp);
      }
      return mlir::WalkResult::advance();
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
  //     cir.resume
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
      rewriter.replaceOpWithNewOp<cir::ResumeOp>(yieldOp);
    } else {
      cleanupOp->emitError("Not yet implemented: cleanup region terminated "
                           "with non-yield operation");
    }

    return clonedEntry;
  }

  // Create a shared unwind destination block for all calls within the same
  // cleanup scope. The unwind block contains a cir.eh.initiate operation
  // (with the cleanup attribute) and a branch to the EH cleanup block.
  mlir::Block *buildUnwindBlock(mlir::Block *ehCleanupBlock, mlir::Location loc,
                                mlir::Block *insertBefore,
                                mlir::PatternRewriter &rewriter) const {
    mlir::Block *unwindBlock = rewriter.createBlock(insertBefore);
    rewriter.setInsertionPointToEnd(unwindBlock);
    auto ehInitiate =
        cir::EhInitiateOp::create(rewriter, loc, /*cleanup=*/true);
    cir::BrOp::create(rewriter, loc, mlir::ValueRange{ehInitiate.getEhToken()},
                      ehCleanupBlock);
    return unwindBlock;
  }

  // Replace a cir.call with a cir.try_call that unwinds to the `unwindDest`
  // block if an exception is thrown.
  void replaceCallWithTryCall(cir::CallOp callOp, mlir::Block *unwindDest,
                              mlir::Location loc,
                              mlir::PatternRewriter &rewriter) const {
    mlir::Block *callBlock = callOp->getBlock();

    assert(!callOp.getNothrow() && "call is not expected to throw");

    // Split the block after the call - remaining ops become the normal
    // destination.
    mlir::Block *normalDest =
        rewriter.splitBlock(callBlock, std::next(callOp->getIterator()));

    // Build the try_call to replace the original call.
    rewriter.setInsertionPoint(callOp);
    mlir::Type resType = callOp->getNumResults() > 0
                             ? callOp->getResult(0).getType()
                             : mlir::Type();
    auto tryCallOp =
        cir::TryCallOp::create(rewriter, loc, callOp.getCalleeAttr(), resType,
                               normalDest, unwindDest, callOp.getArgOperands());

    // Replace uses of the call result with the try_call result.
    if (callOp->getNumResults() > 0)
      callOp->getResult(0).replaceAllUsesWith(tryCallOp.getResult());

    rewriter.eraseOp(callOp);
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
    // is: body -> normal cleanup -> exit ->unwind -> EH cleanup -> continue.
    // If there are no throwing calls, we don't need to EH cleanup blocks.
    mlir::Block *unwindBlock = nullptr;
    mlir::Block *ehCleanupEntry = nullptr;
    if (hasEHCleanup && !callsToRewrite.empty()) {
      ehCleanupEntry =
          buildEHCleanupBlocks(cleanupOp, loc, continueBlock, rewriter);
      unwindBlock =
          buildUnwindBlock(ehCleanupEntry, loc, ehCleanupEntry, rewriter);
    }

    // All normal flow blocks are inserted before this point — either before
    // the unwind block (if EH cleanup exists) or before the continue block.
    mlir::Block *normalInsertPt = unwindBlock ? unwindBlock : continueBlock;

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

    // Erase the original cleanup scope op.
    rewriter.eraseOp(cleanupOp);

    return result;
  }

  mlir::LogicalResult
  matchAndRewrite(cir::CleanupScopeOp cleanupOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    // Nested cleanup scopes must be lowered before the enclosing cleanup scope.
    // Fail the match so the pattern rewriter will process inner cleanups first.
    bool hasNestedCleanup = cleanupOp.getBodyRegion()
                                .walk([&](cir::CleanupScopeOp) {
                                  return mlir::WalkResult::interrupt();
                                })
                                .wasInterrupted();
    if (hasNestedCleanup)
      return mlir::failure();

    cir::CleanupKind cleanupKind = cleanupOp.getCleanupKind();

    // EH cleanups nested inside another cleanup scope are not yet supported
    // because the inner EH unwind path must chain through the outer cleanup
    // before unwinding to the caller.
    if (cleanupKind != cir::CleanupKind::Normal) {
      if (cleanupOp->getParentOfType<cir::CleanupScopeOp>())
        return cleanupOp->emitError(
            "nested EH cleanup scope flattening is not yet implemented");
    }

    // Throwing calls in the cleanup region of an EH-enabled cleanup scope
    // are not yet supported. Such calls would need their own EH handling
    // (e.g., terminate or nested cleanup) during the unwind path.
    if (cleanupKind != cir::CleanupKind::Normal) {
      llvm::SmallVector<cir::CallOp> cleanupThrowingCalls;
      collectThrowingCalls(cleanupOp.getCleanupRegion(), cleanupThrowingCalls);
      if (!cleanupThrowingCalls.empty())
        return cleanupOp->emitError(
            "throwing calls in cleanup region are not yet implemented");
    }

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

    return flattenCleanup(cleanupOp, exits, callsToRewrite, rewriter);
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
    // Cleanup scopes must be lowered before the enclosing try so that
    // EH cleanup inside them is properly handled.
    // Fail the match so the pattern rewriter will process cleanup scopes first.
    bool hasNestedCleanup = tryOp
                                ->walk([&](cir::CleanupScopeOp) {
                                  return mlir::WalkResult::interrupt();
                                })
                                .wasInterrupted();
    if (hasNestedCleanup)
      return mlir::failure();

    mlir::ArrayAttr handlers = tryOp.getHandlerTypesAttr();
    if (handlers && !handlers.empty())
      return tryOp->emitError(
          "TryOp flattening with handlers is not yet implemented");

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

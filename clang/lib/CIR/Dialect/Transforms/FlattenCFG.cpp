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
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/IR/Block.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Support/LogicalResult.h"
#include "aiir/Transforms/DialectConversion.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace aiir;
using namespace cir;

namespace aiir {
#define GEN_PASS_DEF_CIRFLATTENCFG
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace aiir

namespace {

/// Lowers operations with the terminator trait that have a single successor.
void lowerTerminator(aiir::Operation *op, aiir::Block *dest,
                     aiir::PatternRewriter &rewriter) {
  assert(op->hasTrait<aiir::OpTrait::IsTerminator>() && "not a terminator");
  aiir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<cir::BrOp>(op, dest);
}

/// Walks a region while skipping operations of type `Ops`. This ensures the
/// callback is not applied to said operations and its children.
template <typename... Ops>
void walkRegionSkipping(
    aiir::Region &region,
    aiir::function_ref<aiir::WalkResult(aiir::Operation *)> callback) {
  region.walk<aiir::WalkOrder::PreOrder>([&](aiir::Operation *op) {
    if (isa<Ops...>(op))
      return aiir::WalkResult::skip();
    return callback(op);
  });
}

struct CIRFlattenCFGPass : public impl::CIRFlattenCFGBase<CIRFlattenCFGPass> {

  CIRFlattenCFGPass() = default;
  void runOnOperation() override;
};

struct CIRIfFlattening : public aiir::OpRewritePattern<cir::IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(cir::IfOp ifOp,
                  aiir::PatternRewriter &rewriter) const override {
    aiir::OpBuilder::InsertionGuard guard(rewriter);
    aiir::Location loc = ifOp.getLoc();
    bool emptyElse = ifOp.getElseRegion().empty();
    aiir::Block *currentBlock = rewriter.getInsertionBlock();
    aiir::Block *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    aiir::Block *continueBlock;
    if (ifOp->getResults().empty())
      continueBlock = remainingOpsBlock;
    else
      llvm_unreachable("NYI");

    // Inline the region
    aiir::Block *thenBeforeBody = &ifOp.getThenRegion().front();
    aiir::Block *thenAfterBody = &ifOp.getThenRegion().back();
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), continueBlock);

    rewriter.setInsertionPointToEnd(thenAfterBody);
    if (auto thenYieldOp =
            dyn_cast<cir::YieldOp>(thenAfterBody->getTerminator())) {
      rewriter.replaceOpWithNewOp<cir::BrOp>(thenYieldOp, thenYieldOp.getArgs(),
                                             continueBlock);
    }

    rewriter.setInsertionPointToEnd(continueBlock);

    // Has else region: inline it.
    aiir::Block *elseBeforeBody = nullptr;
    aiir::Block *elseAfterBody = nullptr;
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
    return aiir::success();
  }
};

class CIRScopeOpFlattening : public aiir::OpRewritePattern<cir::ScopeOp> {
public:
  using OpRewritePattern<cir::ScopeOp>::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(cir::ScopeOp scopeOp,
                  aiir::PatternRewriter &rewriter) const override {
    aiir::OpBuilder::InsertionGuard guard(rewriter);
    aiir::Location loc = scopeOp.getLoc();

    // Empty scope: just remove it.
    // TODO: Remove this logic once CIR uses AIIR infrastructure to remove
    // trivially dead operations. AIIR canonicalizer is too aggressive and we
    // need to either (a) make sure all our ops model all side-effects and/or
    // (b) have more options in the canonicalizer in AIIR to temper
    // aggressiveness level.
    if (scopeOp.isEmpty()) {
      rewriter.eraseOp(scopeOp);
      return aiir::success();
    }

    // Split the current block before the ScopeOp to create the inlining
    // point.
    aiir::Block *currentBlock = rewriter.getInsertionBlock();
    aiir::Block *continueBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    if (scopeOp.getNumResults() > 0)
      continueBlock->addArguments(scopeOp.getResultTypes(), loc);

    // Inline body region.
    aiir::Block *beforeBody = &scopeOp.getScopeRegion().front();
    aiir::Block *afterBody = &scopeOp.getScopeRegion().back();
    rewriter.inlineRegionBefore(scopeOp.getScopeRegion(), continueBlock);

    // Save stack and then branch into the body of the region.
    rewriter.setInsertionPointToEnd(currentBlock);
    assert(!cir::MissingFeatures::stackSaveOp());
    cir::BrOp::create(rewriter, loc, aiir::ValueRange(), beforeBody);

    // Replace the scopeop return with a branch that jumps out of the body.
    // Stack restore before leaving the body region.
    rewriter.setInsertionPointToEnd(afterBody);
    if (auto yieldOp = dyn_cast<cir::YieldOp>(afterBody->getTerminator())) {
      rewriter.replaceOpWithNewOp<cir::BrOp>(yieldOp, yieldOp.getArgs(),
                                             continueBlock);
    }

    // Replace the op with values return from the body region.
    rewriter.replaceOp(scopeOp, continueBlock->getArguments());

    return aiir::success();
  }
};

class CIRSwitchOpFlattening : public aiir::OpRewritePattern<cir::SwitchOp> {
public:
  using OpRewritePattern<cir::SwitchOp>::OpRewritePattern;

  inline void rewriteYieldOp(aiir::PatternRewriter &rewriter,
                             cir::YieldOp yieldOp,
                             aiir::Block *destination) const {
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<cir::BrOp>(yieldOp, yieldOp.getOperands(),
                                           destination);
  }

  // Return the new defaultDestination block.
  Block *condBrToRangeDestination(cir::SwitchOp op,
                                  aiir::PatternRewriter &rewriter,
                                  aiir::Block *rangeDestination,
                                  aiir::Block *defaultDestination,
                                  const APInt &lowerBound,
                                  const APInt &upperBound) const {
    assert(lowerBound.sle(upperBound) && "Invalid range");
    aiir::Block *resBlock = rewriter.createBlock(defaultDestination);
    cir::IntType sIntType = cir::IntType::get(op.getContext(), 32, true);
    cir::IntType uIntType = cir::IntType::get(op.getContext(), 32, false);

    cir::ConstantOp rangeLength = cir::ConstantOp::create(
        rewriter, op.getLoc(),
        cir::IntAttr::get(sIntType, upperBound - lowerBound));

    cir::ConstantOp lowerBoundValue = cir::ConstantOp::create(
        rewriter, op.getLoc(), cir::IntAttr::get(sIntType, lowerBound));
    aiir::Value diffValue = cir::SubOp::create(
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

  aiir::LogicalResult
  matchAndRewrite(cir::SwitchOp op,
                  aiir::PatternRewriter &rewriter) const override {
    // Cleanup scopes must be lowered before the enclosing switch so that
    // break inside them is properly routed through cleanup.
    // Fail the match so the pattern rewriter will process cleanup scopes first.
    bool hasNestedCleanup = op->walk([&](cir::CleanupScopeOp) {
                                return aiir::WalkResult::interrupt();
                              }).wasInterrupted();
    if (hasNestedCleanup)
      return aiir::failure();

    llvm::SmallVector<CaseOp> cases;
    op.collectCases(cases);

    // Empty switch statement: just erase it.
    if (cases.empty()) {
      rewriter.eraseOp(op);
      return aiir::success();
    }

    // Create exit block from the next node of cir.switch op.
    aiir::Block *exitBlock = rewriter.splitBlock(
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
      for (aiir::Block &block :
           llvm::make_early_inc_range(op.getBody().getBlocks()))
        if (auto yieldOp = dyn_cast<cir::YieldOp>(block.getTerminator()))
          switchYield = yieldOp;

      assert(!op.getBody().empty());
      aiir::Block *originalBlock = op->getBlock();
      aiir::Block *swopBlock =
          rewriter.splitBlock(originalBlock, op->getIterator());
      rewriter.inlineRegionBefore(op.getBody(), exitBlock);

      if (switchYield)
        rewriteYieldOp(rewriter, switchYield, exitBlock);

      rewriter.setInsertionPointToEnd(originalBlock);
      cir::BrOp::create(rewriter, op.getLoc(), swopBlock);
    }

    // Allocate required data structures (disconsider default case in
    // vectors).
    llvm::SmallVector<aiir::APInt, 8> caseValues;
    llvm::SmallVector<aiir::Block *, 8> caseDestinations;
    llvm::SmallVector<aiir::ValueRange, 8> caseOperands;

    llvm::SmallVector<std::pair<APInt, APInt>> rangeValues;
    llvm::SmallVector<aiir::Block *> rangeDestinations;
    llvm::SmallVector<aiir::ValueRange> rangeOperands;

    // Initialize default case as optional.
    aiir::Block *defaultDestination = exitBlock;
    aiir::ValueRange defaultOperands = exitBlock->getArguments();

    // Digest the case statements values and bodies.
    for (cir::CaseOp caseOp : cases) {
      aiir::Region &region = caseOp.getCaseRegion();

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
        for (const aiir::Attribute &value : caseOp.getValue()) {
          caseValues.push_back(cast<cir::IntAttr>(value).getValue());
          caseDestinations.push_back(&region.front());
          caseOperands.push_back(caseDestinations.back()->getArguments());
        }
        break;
      }

      // Handle break statements.
      walkRegionSkipping<cir::LoopOpInterface, cir::SwitchOp>(
          region, [&](aiir::Operation *op) {
            if (!isa<cir::BreakOp>(op))
              return aiir::WalkResult::advance();

            lowerTerminator(op, exitBlock, rewriter);
            return aiir::WalkResult::skip();
          });

      // Track fallthrough in cases.
      for (aiir::Block &blk : region.getBlocks()) {
        if (blk.getNumSuccessors())
          continue;

        if (auto yieldOp = dyn_cast<cir::YieldOp>(blk.getTerminator())) {
          aiir::Operation *nextOp = caseOp->getNextNode();
          assert(nextOp && "caseOp is not expected to be the last op");
          aiir::Block *oldBlock = nextOp->getBlock();
          aiir::Block *newBlock =
              rewriter.splitBlock(oldBlock, nextOp->getIterator());
          rewriter.setInsertionPointToEnd(oldBlock);
          cir::BrOp::create(rewriter, nextOp->getLoc(), aiir::ValueRange(),
                            newBlock);
          rewriteYieldOp(rewriter, yieldOp, newBlock);
        }
      }

      aiir::Block *oldBlock = caseOp->getBlock();
      aiir::Block *newBlock =
          rewriter.splitBlock(oldBlock, caseOp->getIterator());

      aiir::Block &entryBlock = caseOp.getCaseRegion().front();
      rewriter.inlineRegionBefore(caseOp.getCaseRegion(), newBlock);

      // Create a branch to the entry of the inlined region.
      rewriter.setInsertionPointToEnd(oldBlock);
      cir::BrOp::create(rewriter, caseOp.getLoc(), &entryBlock);
    }

    // Remove all cases since we've inlined the regions.
    for (cir::CaseOp caseOp : cases) {
      aiir::Block *caseBlock = caseOp->getBlock();
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

    return aiir::success();
  }
};

class CIRLoopOpInterfaceFlattening
    : public aiir::OpInterfaceRewritePattern<cir::LoopOpInterface> {
public:
  using aiir::OpInterfaceRewritePattern<
      cir::LoopOpInterface>::OpInterfaceRewritePattern;

  inline void lowerConditionOp(cir::ConditionOp op, aiir::Block *body,
                               aiir::Block *exit,
                               aiir::PatternRewriter &rewriter) const {
    aiir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<cir::BrCondOp>(op, op.getCondition(), body,
                                               exit);
  }

  aiir::LogicalResult
  matchAndRewrite(cir::LoopOpInterface op,
                  aiir::PatternRewriter &rewriter) const final {
    // Cleanup scopes must be lowered before the enclosing loop so that
    // break/continue inside them are properly routed through cleanup.
    // Fail the match so the pattern rewriter will process cleanup scopes first.
    bool hasNestedCleanup = op->walk([&](cir::CleanupScopeOp) {
                                return aiir::WalkResult::interrupt();
                              }).wasInterrupted();
    if (hasNestedCleanup)
      return aiir::failure();

    // Setup CFG blocks.
    aiir::Block *entry = rewriter.getInsertionBlock();
    aiir::Block *exit =
        rewriter.splitBlock(entry, rewriter.getInsertionPoint());
    aiir::Block *cond = &op.getCond().front();
    aiir::Block *body = &op.getBody().front();
    aiir::Block *step =
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
    aiir::Block *dest = (step ? step : cond);
    op.walkBodySkippingNestedLoops([&](aiir::Operation *op) {
      if (!isa<cir::ContinueOp>(op))
        return aiir::WalkResult::advance();

      lowerTerminator(op, dest, rewriter);
      return aiir::WalkResult::skip();
    });

    // Lower break statements.
    walkRegionSkipping<cir::LoopOpInterface, cir::SwitchOp>(
        op.getBody(), [&](aiir::Operation *op) {
          if (!isa<cir::BreakOp>(op))
            return aiir::WalkResult::advance();

          lowerTerminator(op, exit, rewriter);
          return aiir::WalkResult::skip();
        });

    // Lower optional body region yield.
    for (aiir::Block &blk : op.getBody().getBlocks()) {
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
    return aiir::success();
  }
};

class CIRTernaryOpFlattening : public aiir::OpRewritePattern<cir::TernaryOp> {
public:
  using OpRewritePattern<cir::TernaryOp>::OpRewritePattern;

  aiir::LogicalResult
  matchAndRewrite(cir::TernaryOp op,
                  aiir::PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Block *condBlock = rewriter.getInsertionBlock();
    Block::iterator opPosition = rewriter.getInsertionPoint();
    Block *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
    llvm::SmallVector<aiir::Location, 2> locs;
    // Ternary result is optional, make sure to populate the location only
    // when relevant.
    if (op->getResultTypes().size())
      locs.push_back(loc);
    Block *continueBlock =
        rewriter.createBlock(remainingOpsBlock, op->getResultTypes(), locs);
    cir::BrOp::create(rewriter, loc, remainingOpsBlock);

    Region &trueRegion = op.getTrueRegion();
    Block *trueBlock = &trueRegion.front();
    aiir::Operation *trueTerminator = trueRegion.back().getTerminator();
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
      return aiir::failure();
    }
    rewriter.inlineRegionBefore(trueRegion, continueBlock);

    Block *falseBlock = continueBlock;
    Region &falseRegion = op.getFalseRegion();

    falseBlock = &falseRegion.front();
    aiir::Operation *falseTerminator = falseRegion.back().getTerminator();
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
      return aiir::failure();
    }
    rewriter.inlineRegionBefore(falseRegion, continueBlock);

    rewriter.setInsertionPointToEnd(condBlock);
    cir::BrCondOp::create(rewriter, loc, op.getCond(), trueBlock, falseBlock);

    rewriter.replaceOp(op, continueBlock->getArguments());

    // Ok, we're done!
    return aiir::success();
  }
};

// Get or create the cleanup destination slot for a function. This slot is
// shared across all cleanup scopes in the function to track which exit path
// to take after running cleanup code when there are multiple exits.
static cir::AllocaOp getOrCreateCleanupDestSlot(cir::FuncOp funcOp,
                                                aiir::PatternRewriter &rewriter,
                                                aiir::Location loc) {
  aiir::Block &entryBlock = funcOp.getBody().front();

  // Look for an existing cleanup dest slot in the entry block.
  auto it = llvm::find_if(entryBlock, [](auto &op) {
    return aiir::isa<AllocaOp>(&op) &&
           aiir::cast<AllocaOp>(&op).getCleanupDestSlot();
  });
  if (it != entryBlock.end())
    return aiir::cast<cir::AllocaOp>(*it);

  // Create a new cleanup dest slot at the start of the entry block.
  aiir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&entryBlock);
  cir::IntType s32Type =
      cir::IntType::get(rewriter.getContext(), 32, /*isSigned=*/true);
  cir::PointerType ptrToS32Type = cir::PointerType::get(s32Type);
  cir::CIRDataLayout dataLayout(funcOp->getParentOfType<aiir::ModuleOp>());
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
collectThrowingCalls(aiir::Region &region,
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
static void collectResumeOps(aiir::Region &region,
                             llvm::SmallVectorImpl<cir::ResumeOp> &resumeOps) {
  region.walk([&](cir::ResumeOp resumeOp) { resumeOps.push_back(resumeOp); });
}

// Replace a cir.call with a cir.try_call that unwinds to the `unwindDest`
// block if an exception is thrown.
static void replaceCallWithTryCall(cir::CallOp callOp, aiir::Block *unwindDest,
                                   aiir::Location loc,
                                   aiir::PatternRewriter &rewriter) {
  aiir::Block *callBlock = callOp->getBlock();

  assert(!callOp.getNothrow() && "call is not expected to throw");

  // Split the block after the call - remaining ops become the normal
  // destination.
  aiir::Block *normalDest =
      rewriter.splitBlock(callBlock, std::next(callOp->getIterator()));

  // Build the try_call to replace the original call.
  rewriter.setInsertionPoint(callOp);
  cir::TryCallOp tryCallOp;
  if (callOp.isIndirect()) {
    aiir::Value indTarget = callOp.getIndirectCall();
    auto ptrTy = aiir::cast<cir::PointerType>(indTarget.getType());
    auto resTy = aiir::cast<cir::FuncType>(ptrTy.getPointee());
    tryCallOp =
        cir::TryCallOp::create(rewriter, loc, indTarget, resTy, normalDest,
                               unwindDest, callOp.getArgOperands());
  } else {
    aiir::Type resType = callOp->getNumResults() > 0
                             ? callOp->getResult(0).getType()
                             : aiir::Type();
    tryCallOp =
        cir::TryCallOp::create(rewriter, loc, callOp.getCalleeAttr(), resType,
                               normalDest, unwindDest, callOp.getArgOperands());
  }

  // Copy all attributes from the original call except those already set by
  // TryCallOp::create or that are operation-specific and should not be copied.
  llvm::StringRef excludedAttrs[] = {
      CIRDialect::getCalleeAttrName(), // Set by create()
      CIRDialect::getOperandSegmentSizesAttrName(),
  };
#ifndef NDEBUG
  // We don't expect to ever see any of these attributes on a call that we
  // converted to a try_call.
  llvm::StringRef unexpectedAttrs[] = {
      CIRDialect::getNoThrowAttrName(),
      CIRDialect::getNoUnwindAttrName(),
  };
#endif
  for (aiir::NamedAttribute attr : callOp->getAttrs()) {
    if (llvm::is_contained(excludedAttrs, attr.getName()))
      continue;
    assert(!llvm::is_contained(unexpectedAttrs, attr.getName()) &&
           "unexpected attribute on converted call");
    tryCallOp->setAttr(attr.getName(), attr.getValue());
  }

  // Replace uses of the call result with the try_call result.
  if (callOp->getNumResults() > 0)
    callOp->getResult(0).replaceAllUsesWith(tryCallOp.getResult());

  rewriter.eraseOp(callOp);
}

// Create a shared unwind destination block. The block contains a
// cir.eh.initiate operation (optionally with the cleanup attribute) and a
// branch to the given destination block, passing the eh_token.
static aiir::Block *buildUnwindBlock(aiir::Block *dest, bool hasCleanup,
                                     aiir::Location loc,
                                     aiir::Block *insertBefore,
                                     aiir::PatternRewriter &rewriter) {
  aiir::Block *unwindBlock = rewriter.createBlock(insertBefore);
  rewriter.setInsertionPointToEnd(unwindBlock);
  auto ehInitiate =
      cir::EhInitiateOp::create(rewriter, loc, /*cleanup=*/hasCleanup);
  cir::BrOp::create(rewriter, loc, aiir::ValueRange{ehInitiate.getEhToken()},
                    dest);
  return unwindBlock;
}

// Create a shared terminate unwind block for throwing calls in EH cleanup
// regions. When an exception is thrown during cleanup (unwinding), the C++
// standard requires that std::terminate() be called.
static aiir::Block *buildTerminateUnwindBlock(aiir::Location loc,
                                              aiir::Block *insertBefore,
                                              aiir::PatternRewriter &rewriter) {
  aiir::Block *terminateBlock = rewriter.createBlock(insertBefore);
  rewriter.setInsertionPointToEnd(terminateBlock);
  auto ehInitiate = cir::EhInitiateOp::create(rewriter, loc, /*cleanup=*/false);
  cir::EhTerminateOp::create(rewriter, loc, ehInitiate.getEhToken());
  return terminateBlock;
}

class CIRCleanupScopeOpFlattening
    : public aiir::OpRewritePattern<cir::CleanupScopeOp> {
public:
  using OpRewritePattern<cir::CleanupScopeOp>::OpRewritePattern;

  struct CleanupExit {
    // An operation that exits the cleanup scope (yield, break, continue,
    // return, etc.)
    aiir::Operation *exitOp;

    // A unique identifier for this exit's destination (used for switch dispatch
    // when there are multiple exits).
    int destinationId;

    CleanupExit(aiir::Operation *op, int id) : exitOp(op), destinationId(id) {}
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
  void collectExits(aiir::Region &cleanupBodyRegion,
                    llvm::SmallVectorImpl<CleanupExit> &exits,
                    int &nextId) const {
    // Collect yield terminators from the body region. We do this separately
    // because yields in nested operations, including those in nested cleanup
    // scopes, won't branch through the outer cleanup region.
    for (aiir::Block &block : cleanupBodyRegion) {
      auto *terminator = block.getTerminator();
      if (isa<cir::YieldOp>(terminator))
        exits.emplace_back(terminator, nextId++);
    }

    // Lambda to walk a loop and collect only returns and gotos.
    // Break and continue inside loops are handled by the loop itself.
    // Loops don't require special handling for nested switch or cleanup scopes
    // because break and continue never branch out of the loop.
    auto collectExitsInLoop = [&](aiir::Operation *loopOp) {
      loopOp->walk<aiir::WalkOrder::PreOrder>([&](aiir::Operation *nestedOp) {
        if (isa<cir::ReturnOp, cir::GotoOp>(nestedOp))
          exits.emplace_back(nestedOp, nextId++);
        return aiir::WalkResult::advance();
      });
    };

    // Forward declaration for mutual recursion.
    std::function<void(aiir::Region &, bool)> collectExitsInCleanup;
    std::function<void(aiir::Operation *)> collectExitsInSwitch;

    // Lambda to collect exits from a switch. Collects return/goto/continue but
    // not break (handled by switch). For nested loops/cleanups, recurses.
    collectExitsInSwitch = [&](aiir::Operation *switchOp) {
      switchOp->walk<aiir::WalkOrder::PreOrder>([&](aiir::Operation *nestedOp) {
        if (isa<cir::CleanupScopeOp>(nestedOp)) {
          // Walk the nested cleanup, but ignore break statements because they
          // will be handled by the switch we are currently walking.
          collectExitsInCleanup(
              cast<cir::CleanupScopeOp>(nestedOp).getBodyRegion(),
              /*ignoreBreak=*/true);
          return aiir::WalkResult::skip();
        } else if (isa<cir::LoopOpInterface>(nestedOp)) {
          collectExitsInLoop(nestedOp);
          return aiir::WalkResult::skip();
        } else if (isa<cir::ReturnOp, cir::GotoOp, cir::ContinueOp>(nestedOp)) {
          exits.emplace_back(nestedOp, nextId++);
        }
        return aiir::WalkResult::advance();
      });
    };

    // Lambda to collect exits from a cleanup scope body region. This collects
    // break (optionally), continue, return, and goto, handling nested loops,
    // switches, and cleanups appropriately.
    collectExitsInCleanup = [&](aiir::Region &region, bool ignoreBreak) {
      region.walk<aiir::WalkOrder::PreOrder>([&](aiir::Operation *op) {
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
          return aiir::WalkResult::skip();
        } else if (isa<cir::LoopOpInterface>(op)) {
          // This kicks off a separate walk rather than continuing to dig deeper
          // in the current walk because we need to handle break and continue
          // differently inside loops.
          collectExitsInLoop(op);
          return aiir::WalkResult::skip();
        } else if (isa<cir::SwitchOp>(op)) {
          // This kicks off a separate walk rather than continuing to dig deeper
          // in the current walk because we need to handle break differently
          // inside switches.
          collectExitsInSwitch(op);
          return aiir::WalkResult::skip();
        }
        return aiir::WalkResult::advance();
      });
    };

    // Collect exits from the body region.
    collectExitsInCleanup(cleanupBodyRegion, /*ignoreBreak=*/false);
  }

  // Check if an operand's defining op should be moved to the destination block.
  // We only sink constants and simple loads. Anything else should be saved
  // to a temporary alloca and reloaded at the destination block.
  static bool shouldSinkReturnOperand(aiir::Value operand,
                                      cir::ReturnOp returnOp) {
    // Block arguments can't be moved
    aiir::Operation *defOp = operand.getDefiningOp();
    if (!defOp)
      return false;

    // Only move constants and loads to the dispatch block. For anything else,
    // we'll store to a temporary and reload in the dispatch block.
    if (!aiir::isa<cir::ConstantOp, cir::LoadOp>(defOp))
      return false;

    // Check if the return is the only user
    if (!operand.hasOneUse())
      return false;

    // Only move ops that are in the same block as the return.
    if (defOp->getBlock() != returnOp->getBlock())
      return false;

    if (auto loadOp = aiir::dyn_cast<cir::LoadOp>(defOp)) {
      // Only attempt to move loads of allocas in the entry block.
      aiir::Value ptr = loadOp.getAddr();
      auto funcOp = returnOp->getParentOfType<cir::FuncOp>();
      assert(funcOp && "Return op has no function parent?");
      aiir::Block &funcEntryBlock = funcOp.getBody().front();

      // Check if it's an alloca in the function entry block
      if (auto allocaOp =
              aiir::dyn_cast_if_present<cir::AllocaOp>(ptr.getDefiningOp()))
        return allocaOp->getBlock() == &funcEntryBlock;

      return false;
    }

    // Make sure we only fall through to here with constants.
    assert(aiir::isa<cir::ConstantOp>(defOp) && "Expected constant op");
    return true;
  }

  // For returns with operands in cleanup dispatch blocks, the operands may not
  // dominate the dispatch block. This function handles that by either sinking
  // the operand's defining op to the dispatch block (for constants and simple
  // loads) or by storing to a temporary alloca and reloading it.
  void
  getReturnOpOperands(cir::ReturnOp returnOp, aiir::Operation *exitOp,
                      aiir::Location loc, aiir::PatternRewriter &rewriter,
                      llvm::SmallVectorImpl<aiir::Value> &returnValues) const {
    aiir::Block *destBlock = rewriter.getInsertionBlock();
    auto funcOp = exitOp->getParentOfType<cir::FuncOp>();
    assert(funcOp && "Return op has no function parent?");
    aiir::Block &funcEntryBlock = funcOp.getBody().front();

    for (aiir::Value operand : returnOp.getOperands()) {
      if (shouldSinkReturnOperand(operand, returnOp)) {
        // Sink the defining op to the dispatch block.
        aiir::Operation *defOp = operand.getDefiningOp();
        defOp->moveBefore(destBlock, destBlock->end());
        returnValues.push_back(operand);
      } else {
        // Create an alloca in the function entry block.
        cir::AllocaOp alloca;
        {
          aiir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&funcEntryBlock);
          cir::CIRDataLayout dataLayout(
              funcOp->getParentOfType<aiir::ModuleOp>());
          uint64_t alignment =
              dataLayout.getAlignment(operand.getType(), true).value();
          cir::PointerType ptrType = cir::PointerType::get(operand.getType());
          alloca = cir::AllocaOp::create(rewriter, loc, ptrType,
                                         operand.getType(), "__ret_operand_tmp",
                                         rewriter.getI64IntegerAttr(alignment));
        }

        // Store the operand value at the original return location.
        {
          aiir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(exitOp);
          cir::StoreOp::create(rewriter, loc, operand, alloca,
                               /*isVolatile=*/false,
                               /*alignment=*/aiir::IntegerAttr(),
                               cir::SyncScopeKindAttr(), cir::MemOrderAttr());
        }

        // Reload the value from the temporary alloca in the destination block.
        rewriter.setInsertionPointToEnd(destBlock);
        auto loaded = cir::LoadOp::create(
            rewriter, loc, alloca, /*isDeref=*/false,
            /*isVolatile=*/false, /*alignment=*/aiir::IntegerAttr(),
            cir::SyncScopeKindAttr(), cir::MemOrderAttr());
        returnValues.push_back(loaded);
      }
    }
  }

  // Create the appropriate terminator for an exit operation in the dispatch
  // block. For return ops with operands, this handles the dominance issue by
  // either moving the operand's defining op to the dispatch block (if it's a
  // trivial use) or by storing to a temporary alloca and loading it.
  aiir::LogicalResult
  createExitTerminator(aiir::Operation *exitOp, aiir::Location loc,
                       aiir::Block *continueBlock,
                       aiir::PatternRewriter &rewriter) const {
    return llvm::TypeSwitch<aiir::Operation *, aiir::LogicalResult>(exitOp)
        .Case<cir::YieldOp>([&](auto) {
          // Yield becomes a branch to continue block.
          cir::BrOp::create(rewriter, loc, continueBlock);
          return aiir::success();
        })
        .Case<cir::BreakOp>([&](auto) {
          // Break is preserved for later lowering by enclosing switch/loop.
          cir::BreakOp::create(rewriter, loc);
          return aiir::success();
        })
        .Case<cir::ContinueOp>([&](auto) {
          // Continue is preserved for later lowering by enclosing loop.
          cir::ContinueOp::create(rewriter, loc);
          return aiir::success();
        })
        .Case<cir::ReturnOp>([&](auto returnOp) {
          // Return from the cleanup exit. Note, if this is a return inside a
          // nested cleanup scope, the flattening of the outer scope will handle
          // branching through the outer cleanup.
          if (returnOp.hasOperand()) {
            llvm::SmallVector<aiir::Value, 2> returnValues;
            getReturnOpOperands(returnOp, exitOp, loc, rewriter, returnValues);
            cir::ReturnOp::create(rewriter, loc, returnValues);
          } else {
            cir::ReturnOp::create(rewriter, loc);
          }
          return aiir::success();
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
        .Default([&](aiir::Operation *op) {
          cir::UnreachableOp::create(rewriter, loc);
          return op->emitError(
              "unexpected exit operation in cleanup scope body");
        });
  }

#ifndef NDEBUG
  // Check that no block other than the last one in a region exits the region.
  static bool regionExitsOnlyFromLastBlock(aiir::Region &region) {
    for (aiir::Block &block : region) {
      if (&block == &region.back())
        continue;
      bool expectedTerminator =
          llvm::TypeSwitch<aiir::Operation *, bool>(block.getTerminator())
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
              .Default([](aiir::Operation *) -> bool {
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
  aiir::Block *buildEHCleanupBlocks(cir::CleanupScopeOp cleanupOp,
                                    aiir::Location loc,
                                    aiir::Block *insertBefore,
                                    aiir::PatternRewriter &rewriter) const {
    assert(regionExitsOnlyFromLastBlock(cleanupOp.getCleanupRegion()) &&
           "cleanup region has exits in non-final blocks");

    // Track the block before the insertion point so we can find the cloned
    // blocks after cloning.
    aiir::Block *blockBeforeClone = insertBefore->getPrevNode();

    // Clone the entire cleanup region before insertBefore.
    rewriter.cloneRegionBefore(cleanupOp.getCleanupRegion(), insertBefore);

    // Find the first cloned block.
    aiir::Block *clonedEntry = blockBeforeClone
                                   ? blockBeforeClone->getNextNode()
                                   : &insertBefore->getParent()->front();

    // Add the eh_token argument to the cloned entry block and insert
    // begin_cleanup at the top.
    auto ehTokenType = cir::EhTokenType::get(rewriter.getContext());
    aiir::Value ehToken = clonedEntry->addArgument(ehTokenType, loc);

    rewriter.setInsertionPointToStart(clonedEntry);
    auto beginCleanup = cir::BeginCleanupOp::create(rewriter, loc, ehToken);

    // Replace the yield terminator in the last cloned block with
    // end_cleanup + resume.
    aiir::Block *lastClonedBlock = insertBefore->getPrevNode();
    auto yieldOp =
        aiir::dyn_cast<cir::YieldOp>(lastClonedBlock->getTerminator());
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
  aiir::LogicalResult
  flattenCleanup(cir::CleanupScopeOp cleanupOp,
                 llvm::SmallVectorImpl<CleanupExit> &exits,
                 llvm::SmallVectorImpl<cir::CallOp> &callsToRewrite,
                 llvm::SmallVectorImpl<cir::ResumeOp> &resumeOpsToChain,
                 aiir::PatternRewriter &rewriter) const {
    aiir::Location loc = cleanupOp.getLoc();
    cir::CleanupKind cleanupKind = cleanupOp.getCleanupKind();
    bool hasNormalCleanup = cleanupKind == cir::CleanupKind::Normal ||
                            cleanupKind == cir::CleanupKind::All;
    bool hasEHCleanup = cleanupKind == cir::CleanupKind::EH ||
                        cleanupKind == cir::CleanupKind::All;
    bool isMultiExit = exits.size() > 1;

    // Get references to region blocks before inlining.
    aiir::Block *bodyEntry = &cleanupOp.getBodyRegion().front();
    aiir::Block *cleanupEntry = &cleanupOp.getCleanupRegion().front();
    aiir::Block *cleanupExit = &cleanupOp.getCleanupRegion().back();
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
    aiir::Block *currentBlock = rewriter.getInsertionBlock();
    aiir::Block *continueBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

    // Build EH cleanup blocks if needed. This must be done before inlining
    // the cleanup region since buildEHCleanupBlocks clones from it. The unwind
    // block is inserted before the EH cleanup entry so that the final layout
    // is: body -> normal cleanup -> exit -> unwind -> EH cleanup -> continue.
    // EH cleanup blocks are needed when there are throwing calls that need to
    // be rewritten to try_call, or when there are resume ops from
    // already-flattened inner cleanup scopes that need to chain through this
    // cleanup's EH handler.
    aiir::Block *unwindBlock = nullptr;
    aiir::Block *ehCleanupEntry = nullptr;
    if (hasEHCleanup &&
        (!callsToRewrite.empty() || !resumeOpsToChain.empty())) {
      ehCleanupEntry =
          buildEHCleanupBlocks(cleanupOp, loc, continueBlock, rewriter);
      // The unwind block is only needed when there are throwing calls that
      // need a shared unwind destination. Resume ops from inner cleanups
      // branch directly to the EH cleanup entry.
      if (!callsToRewrite.empty())
        unwindBlock = buildUnwindBlock(ehCleanupEntry, /*hasCleanup=*/true, loc,
                                       ehCleanupEntry, rewriter);
    }

    // All normal flow blocks are inserted before this point — either before
    // the unwind block (if it exists), or before the EH cleanup entry (if EH
    // cleanup exists but no unwind block is needed), or before the continue
    // block.
    aiir::Block *normalInsertPt =
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
    aiir::LogicalResult result = aiir::success();
    if (hasNormalCleanup) {
      // Create the exit/dispatch block (after cleanup, before continue).
      aiir::Block *exitBlock = rewriter.createBlock(normalInsertPt);

      // Rewrite the cleanup region's yield to branch to exit block.
      rewriter.setInsertionPoint(cleanupYield);
      rewriter.replaceOpWithNewOp<cir::BrOp>(cleanupYield, exitBlock);

      if (isMultiExit) {
        // Build the dispatch switch in the exit block.
        rewriter.setInsertionPointToEnd(exitBlock);

        // Load the destination slot value.
        auto slotValue = cir::LoadOp::create(
            rewriter, loc, destSlot, /*isDeref=*/false,
            /*isVolatile=*/false, /*alignment=*/aiir::IntegerAttr(),
            cir::SyncScopeKindAttr(), cir::MemOrderAttr());

        // Create destination blocks for each exit and collect switch case info.
        llvm::SmallVector<aiir::APInt, 8> caseValues;
        llvm::SmallVector<aiir::Block *, 8> caseDestinations;
        llvm::SmallVector<aiir::ValueRange, 8> caseOperands;
        cir::IntType s32Type =
            cir::IntType::get(rewriter.getContext(), 32, /*isSigned=*/true);

        for (const CleanupExit &exit : exits) {
          // Create a block for this destination.
          aiir::Block *destBlock = rewriter.createBlock(normalInsertPt);
          rewriter.setInsertionPointToEnd(destBlock);
          result =
              createExitTerminator(exit.exitOp, loc, continueBlock, rewriter);

          // Add to switch cases.
          caseValues.push_back(
              llvm::APInt(32, static_cast<uint64_t>(exit.destinationId), true));
          caseDestinations.push_back(destBlock);
          caseOperands.push_back(aiir::ValueRange());

          // Replace the original exit op with: store dest ID, branch to
          // cleanup.
          rewriter.setInsertionPoint(exit.exitOp);
          auto destIdConst = cir::ConstantOp::create(
              rewriter, loc, cir::IntAttr::get(s32Type, exit.destinationId));
          cir::StoreOp::create(rewriter, loc, destIdConst, destSlot,
                               /*isVolatile=*/false,
                               /*alignment=*/aiir::IntegerAttr(),
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
        aiir::Block *defaultBlock = rewriter.createBlock(normalInsertPt);
        rewriter.setInsertionPointToEnd(defaultBlock);
        cir::UnreachableOp::create(rewriter, loc);

        // Build the switch.flat operation in the exit block.
        rewriter.setInsertionPointToEnd(exitBlock);
        cir::SwitchFlatOp::create(rewriter, loc, slotValue, defaultBlock,
                                  aiir::ValueRange(), caseValues,
                                  caseDestinations, caseOperands);
      } else {
        // Single exit: put the appropriate terminator directly in the exit
        // block.
        rewriter.setInsertionPointToEnd(exitBlock);
        aiir::Operation *exitOp = exits[0].exitOp;
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
      for (aiir::Block *block = ehCleanupEntry; block != continueBlock;
           block = block->getNextNode()) {
        block->walk([&](cir::CallOp callOp) {
          if (!callOp.getNothrow())
            ehCleanupThrowingCalls.push_back(callOp);
        });
      }
      if (!ehCleanupThrowingCalls.empty()) {
        aiir::Block *terminateBlock =
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
        aiir::Value ehToken = resumeOp.getEhToken();
        rewriter.setInsertionPoint(resumeOp);
        rewriter.replaceOpWithNewOp<cir::BrOp>(
            resumeOp, aiir::ValueRange{ehToken}, ehCleanupEntry);
      }
    }

    // Erase the original cleanup scope op.
    rewriter.eraseOp(cleanupOp);

    return result;
  }

  aiir::LogicalResult
  matchAndRewrite(cir::CleanupScopeOp cleanupOp,
                  aiir::PatternRewriter &rewriter) const override {
    aiir::OpBuilder::InsertionGuard guard(rewriter);

    // Nested cleanup scopes and try operations must be flattened before the
    // enclosing cleanup scope so that EH cleanup inside them is properly
    // handled. Fail the match so the pattern rewriter processes them first.
    bool hasNestedOps = cleanupOp.getBodyRegion()
                            .walk([&](aiir::Operation *op) {
                              if (isa<cir::CleanupScopeOp, cir::TryOp>(op))
                                return aiir::WalkResult::interrupt();
                              return aiir::WalkResult::advance();
                            })
                            .wasInterrupted();
    if (hasNestedOps)
      return aiir::failure();

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

class CIRTryOpFlattening : public aiir::OpRewritePattern<cir::TryOp> {
public:
  using OpRewritePattern<cir::TryOp>::OpRewritePattern;

  // Build the catch dispatch block with a cir.eh.dispatch operation.
  // The dispatch block receives an !cir.eh_token argument and dispatches
  // to the appropriate catch handler blocks based on exception types.
  aiir::Block *buildCatchDispatchBlock(
      cir::TryOp tryOp, aiir::ArrayAttr handlerTypes,
      llvm::SmallVectorImpl<aiir::Block *> &catchHandlerBlocks,
      aiir::Location loc, aiir::Block *insertBefore,
      aiir::PatternRewriter &rewriter) const {
    aiir::Block *dispatchBlock = rewriter.createBlock(insertBefore);
    auto ehTokenType = cir::EhTokenType::get(rewriter.getContext());
    aiir::Value ehToken = dispatchBlock->addArgument(ehTokenType, loc);

    rewriter.setInsertionPointToEnd(dispatchBlock);

    // Build the catch types and destinations for the dispatch.
    llvm::SmallVector<aiir::Attribute> catchTypeAttrs;
    llvm::SmallVector<aiir::Block *> catchDests;
    aiir::Block *defaultDest = nullptr;
    bool defaultIsCatchAll = false;

    for (auto [typeAttr, handlerBlock] :
         llvm::zip(handlerTypes, catchHandlerBlocks)) {
      if (aiir::isa<cir::CatchAllAttr>(typeAttr)) {
        assert(!defaultDest && "multiple catch_all or unwind handlers");
        defaultDest = handlerBlock;
        defaultIsCatchAll = true;
      } else if (aiir::isa<cir::UnwindAttr>(typeAttr)) {
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

    aiir::ArrayAttr catchTypesArrayAttr;
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
  aiir::Block *flattenCatchHandler(aiir::Region &handlerRegion,
                                   aiir::Block *continueBlock,
                                   aiir::Location loc,
                                   aiir::Block *insertBefore,
                                   aiir::PatternRewriter &rewriter) const {
    // The handler region entry block has the !cir.eh_token argument.
    aiir::Block *handlerEntry = &handlerRegion.front();

    // Inline the handler region before insertBefore.
    rewriter.inlineRegionBefore(handlerRegion, insertBefore);

    // Replace yield terminators in the handler with branches to continue.
    for (aiir::Block &block : llvm::make_range(handlerEntry->getIterator(),
                                               insertBefore->getIterator())) {
      if (auto yieldOp = dyn_cast<cir::YieldOp>(block.getTerminator())) {
        // Verify that end_catch is the last non-branch operation before
        // this yield. After cleanup scope flattening, end_catch may be in
        // a predecessor block rather than immediately before the yield.
        // Walk back through the single-predecessor chain, verifying that
        // each intermediate block contains only a branch terminator, until
        // we find end_catch as the last non-terminator in some block.
        assert([&]() {
          // Check if end_catch immediately precedes the yield.
          if (aiir::Operation *prev = yieldOp->getPrevNode())
            return isa<cir::EndCatchOp>(prev);
          // The yield is alone in its block. Walk backward through
          // single-predecessor blocks that contain only a branch.
          aiir::Block *b = block.getSinglePredecessor();
          while (b) {
            aiir::Operation *term = b->getTerminator();
            if (aiir::Operation *prev = term->getPrevNode())
              return isa<cir::EndCatchOp>(prev);
            if (!isa<cir::BrOp>(term))
              return false;
            b = b->getSinglePredecessor();
          }
          return false;
        }() && "expected end_catch as last operation before yield "
               "in catch handler, with only branches in between");
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
  aiir::Block *flattenUnwindHandler(aiir::Region &unwindRegion,
                                    aiir::Location loc,
                                    aiir::Block *insertBefore,
                                    aiir::PatternRewriter &rewriter) const {
    aiir::Block *unwindEntry = &unwindRegion.front();
    rewriter.inlineRegionBefore(unwindRegion, insertBefore);
    return unwindEntry;
  }

  aiir::LogicalResult
  matchAndRewrite(cir::TryOp tryOp,
                  aiir::PatternRewriter &rewriter) const override {
    // Nested try ops and cleanup scopes must be flattened before the enclosing
    // try so that EH cleanup inside them is properly handled. Fail the match so
    // the pattern rewriter will process nested ops first.
    bool hasNestedOps =
        tryOp
            ->walk([&](aiir::Operation *op) {
              if (isa<cir::CleanupScopeOp, cir::TryOp>(op) && op != tryOp)
                return aiir::WalkResult::interrupt();
              return aiir::WalkResult::advance();
            })
            .wasInterrupted();
    if (hasNestedOps)
      return aiir::failure();

    aiir::OpBuilder::InsertionGuard guard(rewriter);
    aiir::Location loc = tryOp.getLoc();

    aiir::ArrayAttr handlerTypes = tryOp.getHandlerTypesAttr();
    aiir::MutableArrayRef<aiir::Region> handlerRegions =
        tryOp.getHandlerRegions();

    // Collect throwing calls in the try body.
    llvm::SmallVector<cir::CallOp> callsToRewrite;
    collectThrowingCalls(tryOp.getTryRegion(), callsToRewrite);

    // Collect resume ops from already-flattened cleanup scopes in the try body.
    llvm::SmallVector<cir::ResumeOp> resumeOpsToChain;
    collectResumeOps(tryOp.getTryRegion(), resumeOpsToChain);

    // Split the current block and inline the try body.
    aiir::Block *currentBlock = rewriter.getInsertionBlock();
    aiir::Block *continueBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

    // Get references to try body blocks before inlining.
    aiir::Block *bodyEntry = &tryOp.getTryRegion().front();
    aiir::Block *bodyExit = &tryOp.getTryRegion().back();

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
      return aiir::success();
    }

    // If there are no throwing calls and no resume ops from inner cleanup
    // scopes, exceptions cannot reach the catch handlers. Skip handler and
    // dispatch block creation — the handler regions will be dropped when
    // the try op is erased.
    if (callsToRewrite.empty() && resumeOpsToChain.empty()) {
      rewriter.eraseOp(tryOp);
      return aiir::success();
    }

    // Build the catch handler blocks.

    // First, flatten all handler regions and collect the entry blocks.
    llvm::SmallVector<aiir::Block *> catchHandlerBlocks;

    for (const auto &[idx, typeAttr] : llvm::enumerate(handlerTypes)) {
      aiir::Region &handlerRegion = handlerRegions[idx];

      if (aiir::isa<cir::UnwindAttr>(typeAttr)) {
        aiir::Block *unwindEntry =
            flattenUnwindHandler(handlerRegion, loc, continueBlock, rewriter);
        catchHandlerBlocks.push_back(unwindEntry);
      } else {
        aiir::Block *handlerEntry = flattenCatchHandler(
            handlerRegion, continueBlock, loc, continueBlock, rewriter);
        catchHandlerBlocks.push_back(handlerEntry);
      }
    }

    // Build the catch dispatch block.
    aiir::Block *dispatchBlock =
        buildCatchDispatchBlock(tryOp, handlerTypes, catchHandlerBlocks, loc,
                                catchHandlerBlocks.front(), rewriter);

    // Build a block to be the unwind desination for throwing calls and replace
    // the calls with try_call ops. Note that the unwind block created here is
    // something different than the unwind handler that we may have created
    // above. The unwind handler continues unwinding after uncaught exceptions.
    // This is the block that will eventually become the landing pad for invoke
    // instructions.
    bool hasCleanup = tryOp.getCleanup();
    if (!callsToRewrite.empty()) {
      // Create a shared unwind block for all throwing calls.
      aiir::Block *unwindBlock = buildUnwindBlock(dispatchBlock, hasCleanup,
                                                  loc, dispatchBlock, rewriter);

      for (cir::CallOp callOp : callsToRewrite)
        replaceCallWithTryCall(callOp, unwindBlock, loc, rewriter);
    }

    // Chain resume ops from inner cleanup scopes.
    // Resume ops from already-flattened cleanup scopes within the try body
    // should branch to the catch dispatch block instead of unwinding directly.
    for (cir::ResumeOp resumeOp : resumeOpsToChain) {
      aiir::Value ehToken = resumeOp.getEhToken();
      rewriter.setInsertionPoint(resumeOp);
      rewriter.replaceOpWithNewOp<cir::BrOp>(
          resumeOp, aiir::ValueRange{ehToken}, dispatchBlock);
    }

    // Finally, erase the original try op ----
    rewriter.eraseOp(tryOp);

    return aiir::success();
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
  getOperation()->walk<aiir::WalkOrder::PostOrder>([&](Operation *op) {
    if (isa<IfOp, ScopeOp, SwitchOp, LoopOpInterface, TernaryOp, CleanupScopeOp,
            TryOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (applyOpPatternsGreedily(ops, std::move(patterns)).failed())
    signalPassFailure();
}

} // namespace

namespace aiir {

std::unique_ptr<Pass> createCIRFlattenCFGPass() {
  return std::make_unique<CIRFlattenCFGPass>();
}

} // namespace aiir

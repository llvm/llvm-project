//====- FlattenCFG.cpp - Flatten CIR CFG ----------------------------------===//
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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace mlir::cir;

namespace {

/// Lowers operations with the terminator trait that have a single successor.
void lowerTerminator(mlir::Operation *op, mlir::Block *dest,
                     mlir::PatternRewriter &rewriter) {
  assert(op->hasTrait<mlir::OpTrait::IsTerminator>() && "not a terminator");
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(op, dest);
}

/// Walks a region while skipping operations of type `Ops`. This ensures the
/// callback is not applied to said operations and its children.
template <typename... Ops>
void walkRegionSkipping(mlir::Region &region,
                        mlir::function_ref<void(mlir::Operation *)> callback) {
  region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (isa<Ops...>(op))
      return mlir::WalkResult::skip();
    callback(op);
    return mlir::WalkResult::advance();
  });
}

struct FlattenCFGPass : public FlattenCFGBase<FlattenCFGPass> {

  FlattenCFGPass() = default;
  void runOnOperation() override;
};

struct CIRIfFlattening : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::IfOp ifOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto loc = ifOp.getLoc();
    auto emptyElse = ifOp.getElseRegion().empty();

    auto *currentBlock = rewriter.getInsertionBlock();
    auto *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *continueBlock;
    if (ifOp->getResults().size() == 0)
      continueBlock = remainingOpsBlock;
    else
      llvm_unreachable("NYI");

    // Inline then region
    auto *thenBeforeBody = &ifOp.getThenRegion().front();
    auto *thenAfterBody = &ifOp.getThenRegion().back();
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), continueBlock);

    rewriter.setInsertionPointToEnd(thenAfterBody);
    if (auto thenYieldOp =
            dyn_cast<mlir::cir::YieldOp>(thenAfterBody->getTerminator())) {
      rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
          thenYieldOp, thenYieldOp.getArgs(), continueBlock);
    }

    rewriter.setInsertionPointToEnd(continueBlock);

    // Has else region: inline it.
    mlir::Block *elseBeforeBody = nullptr;
    mlir::Block *elseAfterBody = nullptr;
    if (!emptyElse) {
      elseBeforeBody = &ifOp.getElseRegion().front();
      elseAfterBody = &ifOp.getElseRegion().back();
      rewriter.inlineRegionBefore(ifOp.getElseRegion(), thenAfterBody);
    } else {
      elseBeforeBody = elseAfterBody = continueBlock;
    }

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<mlir::cir::BrCondOp>(loc, ifOp.getCondition(),
                                         thenBeforeBody, elseBeforeBody);

    if (!emptyElse) {
      rewriter.setInsertionPointToEnd(elseAfterBody);
      if (auto elseYieldOp =
              dyn_cast<mlir::cir::YieldOp>(elseAfterBody->getTerminator())) {
        rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
            elseYieldOp, elseYieldOp.getArgs(), continueBlock);
      }
    }

    rewriter.replaceOp(ifOp, continueBlock->getArguments());
    return mlir::success();
  }
};

class CIRScopeOpFlattening : public mlir::OpRewritePattern<mlir::cir::ScopeOp> {
public:
  using OpRewritePattern<mlir::cir::ScopeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ScopeOp scopeOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto loc = scopeOp.getLoc();

    // Empty scope: just remove it.
    if (scopeOp.getRegion().empty()) {
      rewriter.eraseOp(scopeOp);
      return mlir::success();
    }

    // Split the current block before the ScopeOp to create the inlining
    // point.
    auto *currentBlock = rewriter.getInsertionBlock();
    auto *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *continueBlock;
    if (scopeOp.getNumResults() == 0)
      continueBlock = remainingOpsBlock;
    else
      llvm_unreachable("NYI");

    // Inline body region.
    auto *beforeBody = &scopeOp.getRegion().front();
    auto *afterBody = &scopeOp.getRegion().back();
    rewriter.inlineRegionBefore(scopeOp.getRegion(), continueBlock);

    // Save stack and then branch into the body of the region.
    rewriter.setInsertionPointToEnd(currentBlock);
    // TODO(CIR): stackSaveOp
    // auto stackSaveOp = rewriter.create<mlir::LLVM::StackSaveOp>(
    //     loc, mlir::LLVM::LLVMPointerType::get(
    //              mlir::IntegerType::get(scopeOp.getContext(), 8)));
    rewriter.create<mlir::cir::BrOp>(loc, mlir::ValueRange(), beforeBody);

    // Replace the scopeop return with a branch that jumps out of the body.
    // Stack restore before leaving the body region.
    rewriter.setInsertionPointToEnd(afterBody);
    if (auto yieldOp =
            dyn_cast<mlir::cir::YieldOp>(afterBody->getTerminator())) {
      rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(yieldOp, yieldOp.getArgs(),
                                                   continueBlock);
    }

    // TODO(cir): stackrestore?

    // Replace the op with values return from the body region.
    rewriter.replaceOp(scopeOp, continueBlock->getArguments());

    return mlir::success();
  }
};

class CIRTryOpFlattening : public mlir::OpRewritePattern<mlir::cir::TryOp> {
public:
  using OpRewritePattern<mlir::cir::TryOp>::OpRewritePattern;

  mlir::Block *buildTypeCase(mlir::PatternRewriter &rewriter, mlir::Region &r,
                             mlir::Block *afterTry,
                             mlir::Type exceptionPtrTy) const {
    YieldOp yieldOp;
    CatchParamOp paramOp;
    r.walk([&](YieldOp op) {
      assert(!yieldOp && "expect to only find one");
      yieldOp = op;
    });
    r.walk([&](CatchParamOp op) {
      assert(!paramOp && "expect to only find one");
      paramOp = op;
    });
    rewriter.inlineRegionBefore(r, afterTry);

    // Rewrite `cir.catch_param` to be scope aware and instead generate:
    // ```
    //   cir.catch_param begin %exception_ptr
    //   ...
    //   cir.catch_param end
    //   cir.br ...
    mlir::Value catchResult = paramOp.getParam();
    assert(catchResult && "expected to be available");
    rewriter.setInsertionPointAfterValue(catchResult);
    auto catchType = catchResult.getType();
    mlir::Block *entryBlock = paramOp->getBlock();
    mlir::Location catchLoc = paramOp.getLoc();
    // Catch handler only gets the exception pointer (selection not needed).
    mlir::Value exceptionPtr =
        entryBlock->addArgument(exceptionPtrTy, paramOp.getLoc());

    rewriter.replaceOpWithNewOp<mlir::cir::CatchParamOp>(
        paramOp, catchType, exceptionPtr,
        mlir::cir::CatchParamKindAttr::get(rewriter.getContext(),
                                           mlir::cir::CatchParamKind::begin));

    rewriter.setInsertionPoint(yieldOp);
    rewriter.create<mlir::cir::CatchParamOp>(
        catchLoc, mlir::Type{}, nullptr,
        mlir::cir::CatchParamKindAttr::get(rewriter.getContext(),
                                           mlir::cir::CatchParamKind::end));

    rewriter.setInsertionPointToEnd(yieldOp->getBlock());
    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(yieldOp, afterTry);
    return entryBlock;
  }

  void buildUnwindCase(mlir::PatternRewriter &rewriter, mlir::Region &r,
                       mlir::Block *unwindBlock) const {
    assert(&r.front() == &r.back() && "only one block expected");
    rewriter.mergeBlocks(&r.back(), unwindBlock);
    auto resume = dyn_cast<mlir::cir::ResumeOp>(unwindBlock->getTerminator());
    assert(resume && "expected 'cir.resume'");
    rewriter.setInsertionPointToEnd(unwindBlock);
    rewriter.replaceOpWithNewOp<mlir::cir::ResumeOp>(
        resume, unwindBlock->getArgument(0), unwindBlock->getArgument(1));
  }

  void buildAllCase(mlir::PatternRewriter &rewriter, mlir::Region &r,
                    mlir::Block *afterTry, mlir::Block *catchAllBlock,
                    mlir::Value exceptionPtr) const {
    YieldOp yieldOp;
    CatchParamOp paramOp;
    r.walk([&](YieldOp op) {
      assert(!yieldOp && "expect to only find one");
      yieldOp = op;
    });
    r.walk([&](CatchParamOp op) {
      assert(!paramOp && "expect to only find one");
      paramOp = op;
    });
    mlir::Block *catchAllStartBB = &r.front();
    rewriter.inlineRegionBefore(r, afterTry);
    rewriter.mergeBlocks(catchAllStartBB, catchAllBlock);

    // Rewrite `cir.catch_param` to be scope aware and instead generate:
    // ```
    //   cir.catch_param begin %exception_ptr
    //   ...
    //   cir.catch_param end
    //   cir.br ...
    mlir::Value catchResult = paramOp.getParam();
    assert(catchResult && "expected to be available");
    rewriter.setInsertionPointAfterValue(catchResult);
    auto catchType = catchResult.getType();
    mlir::Location catchLoc = paramOp.getLoc();
    rewriter.replaceOpWithNewOp<mlir::cir::CatchParamOp>(
        paramOp, catchType, exceptionPtr,
        mlir::cir::CatchParamKindAttr::get(rewriter.getContext(),
                                           mlir::cir::CatchParamKind::begin));

    rewriter.setInsertionPoint(yieldOp);
    rewriter.create<mlir::cir::CatchParamOp>(
        catchLoc, mlir::Type{}, nullptr,
        mlir::cir::CatchParamKindAttr::get(rewriter.getContext(),
                                           mlir::cir::CatchParamKind::end));

    rewriter.setInsertionPointToEnd(yieldOp->getBlock());
    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(yieldOp, afterTry);
  }

  mlir::ArrayAttr collectTypeSymbols(mlir::cir::TryOp tryOp) const {
    mlir::ArrayAttr caseAttrList = tryOp.getCatchTypesAttr();
    llvm::SmallVector<mlir::Attribute, 4> symbolList;

    for (mlir::Attribute caseAttr : caseAttrList) {
      auto typeIdGlobal = dyn_cast<mlir::cir::GlobalViewAttr>(caseAttr);
      if (!typeIdGlobal)
        continue;
      symbolList.push_back(typeIdGlobal.getSymbol());
    }

    // Return an empty attribute instead of an empty list...
    if (symbolList.empty())
      return {};
    return mlir::ArrayAttr::get(caseAttrList.getContext(), symbolList);
  }

  mlir::Block *buildCatchers(mlir::cir::TryOp tryOp,
                             mlir::PatternRewriter &rewriter,
                             mlir::Block *afterBody,
                             mlir::Block *afterTry) const {
    auto loc = tryOp.getLoc();
    // Replace the tryOp return with a branch that jumps out of the body.
    rewriter.setInsertionPointToEnd(afterBody);
    auto tryBodyYield = cast<mlir::cir::YieldOp>(afterBody->getTerminator());

    mlir::Block *beforeCatch = rewriter.getInsertionBlock();
    auto *catchBegin =
        rewriter.splitBlock(beforeCatch, rewriter.getInsertionPoint());
    rewriter.setInsertionPointToEnd(beforeCatch);
    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(tryBodyYield, afterTry);

    // Start the landing pad by getting the inflight exception information.
    rewriter.setInsertionPointToEnd(catchBegin);
    auto exceptionPtrType = mlir::cir::PointerType::get(
        mlir::cir::VoidType::get(rewriter.getContext()));
    auto typeIdType = mlir::cir::IntType::get(getContext(), 32, false);
    mlir::ArrayAttr symlist = collectTypeSymbols(tryOp);
    auto inflightEh = rewriter.create<mlir::cir::EhInflightOp>(
        loc, exceptionPtrType, typeIdType, symlist);
    auto selector = inflightEh.getTypeId();
    auto exceptionPtr = inflightEh.getExceptionPtr();

    // Handle dispatch. In could in theory use a switch, but let's just
    // mimic LLVM more closely since we have no specific thing to achieve
    // doing that (might not play as well with existing optimizers either).
    auto *nextDispatcher =
        rewriter.splitBlock(catchBegin, rewriter.getInsertionPoint());
    rewriter.setInsertionPointToEnd(catchBegin);
    mlir::ArrayAttr caseAttrList = tryOp.getCatchTypesAttr();
    nextDispatcher->addArgument(exceptionPtr.getType(), loc);
    SmallVector<mlir::Value> dispatcherInitOps = {exceptionPtr};
    bool tryOnlyHasCatchAll = caseAttrList.size() == 1 &&
                              isa<mlir::cir::CatchAllAttr>(caseAttrList[0]);
    if (!tryOnlyHasCatchAll) {
      nextDispatcher->addArgument(selector.getType(), loc);
      dispatcherInitOps.push_back(selector);
    }
    rewriter.create<mlir::cir::BrOp>(loc, nextDispatcher, dispatcherInitOps);

    // Fill in dispatcher.
    rewriter.setInsertionPointToEnd(nextDispatcher);
    llvm::MutableArrayRef<mlir::Region> caseRegions = tryOp.getCatchRegions();
    unsigned caseCnt = 0;

    for (mlir::Attribute caseAttr : caseAttrList) {
      if (auto typeIdGlobal = dyn_cast<mlir::cir::GlobalViewAttr>(caseAttr)) {
        auto *previousDispatcher = nextDispatcher;
        auto typeId = rewriter.create<mlir::cir::EhTypeIdOp>(
            loc, typeIdGlobal.getSymbol());
        auto ehPtr = previousDispatcher->getArgument(0);
        auto ehSel = previousDispatcher->getArgument(1);

        auto match = rewriter.create<mlir::cir::CmpOp>(
            loc, mlir::cir::BoolType::get(rewriter.getContext()),
            mlir::cir::CmpOpKind::eq, ehSel, typeId);

        mlir::Block *typeCatchBlock = buildTypeCase(
            rewriter, caseRegions[caseCnt], afterTry, ehPtr.getType());
        nextDispatcher = rewriter.createBlock(afterTry);
        rewriter.setInsertionPointToEnd(previousDispatcher);

        // Next dispatcher gets by default both exception ptr and selector info,
        // but on a catch all we don't need selector info.
        nextDispatcher->addArgument(ehPtr.getType(), loc);
        SmallVector<mlir::Value> nextDispatchOps = {ehPtr};
        if (!isa<mlir::cir::CatchAllAttr>(caseAttrList[caseCnt + 1])) {
          nextDispatcher->addArgument(ehSel.getType(), loc);
          nextDispatchOps.push_back(ehSel);
        }

        rewriter.create<mlir::cir::BrCondOp>(
            loc, match, typeCatchBlock, nextDispatcher, mlir::ValueRange{ehPtr},
            nextDispatchOps);
        rewriter.setInsertionPointToEnd(nextDispatcher);
      } else if (auto catchAll = dyn_cast<mlir::cir::CatchAllAttr>(caseAttr)) {
        // In case the catch(...) is all we got, `nextDispatcher` shall be
        // non-empty.
        assert(nextDispatcher->getArguments().size() == 1 &&
               "expected one block argument");
        auto ehPtr = nextDispatcher->getArgument(0);
        buildAllCase(rewriter, caseRegions[caseCnt], afterTry, nextDispatcher,
                     ehPtr);
        nextDispatcher = nullptr; // No more business in try/catch
      } else if (auto catchUnwind =
                     dyn_cast<mlir::cir::CatchUnwindAttr>(caseAttr)) {
        // assert(nextDispatcher->empty() && "expect empty dispatcher");
        // assert(!nextDispatcher->args_empty() && "expected block argument");
        assert(nextDispatcher->getArguments().size() == 2 &&
               "expected two block argument");
        buildUnwindCase(rewriter, caseRegions[caseCnt], nextDispatcher);
        nextDispatcher = nullptr; // No more business in try/catch
      }
      caseCnt++;
    }

    assert(!nextDispatcher && "no dispatcher available anymore");
    return catchBegin;
  }

  mlir::Block *buildTryBody(mlir::cir::TryOp tryOp,
                            mlir::PatternRewriter &rewriter) const {
    auto loc = tryOp.getLoc();
    // Split the current block before the TryOp to create the inlining
    // point.
    auto *beforeTryScopeBlock = rewriter.getInsertionBlock();
    mlir::Block *afterTry =
        rewriter.splitBlock(beforeTryScopeBlock, rewriter.getInsertionPoint());

    // Inline body region.
    auto *beforeBody = &tryOp.getTryRegion().front();
    rewriter.inlineRegionBefore(tryOp.getTryRegion(), afterTry);

    // Branch into the body of the region.
    rewriter.setInsertionPointToEnd(beforeTryScopeBlock);
    rewriter.create<mlir::cir::BrOp>(loc, mlir::ValueRange(), beforeBody);
    return afterTry;
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::TryOp tryOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto *afterBody = &tryOp.getTryRegion().back();

    // Empty scope: just remove it.
    if (tryOp.getTryRegion().empty()) {
      rewriter.eraseOp(tryOp);
      return mlir::success();
    }

    // Grab the collection of `cir.call exception`s to rewrite to
    // `cir.try_call`.
    SmallVector<mlir::cir::CallOp, 4> callsToRewrite;
    tryOp.getTryRegion().walk([&](CallOp op) {
      // Only grab calls within immediate closest TryOp scope.
      if (op->getParentOfType<mlir::cir::TryOp>() != tryOp)
        return;
      if (!op.getException())
        return;
      callsToRewrite.push_back(op);
    });

    // Build try body.
    mlir::Block *afterTry = buildTryBody(tryOp, rewriter);

    // Build catchers.
    mlir::Block *landingPad =
        buildCatchers(tryOp, rewriter, afterBody, afterTry);
    rewriter.eraseOp(tryOp);

    // Rewrite calls.
    for (CallOp callOp : callsToRewrite) {
      mlir::Block *callBlock = callOp->getBlock();
      mlir::Block *cont =
          rewriter.splitBlock(callBlock, mlir::Block::iterator(callOp));
      mlir::cir::ExtraFuncAttributesAttr extraAttrs = callOp.getExtraAttrs();
      std::optional<mlir::cir::ASTCallExprInterface> ast = callOp.getAst();

      mlir::FlatSymbolRefAttr symbol;
      if (!callOp.isIndirect())
        symbol = callOp.getCalleeAttr();
      rewriter.setInsertionPointToEnd(callBlock);
      auto tryCall = rewriter.replaceOpWithNewOp<mlir::cir::TryCallOp>(
          callOp, symbol, callOp.getResult().getType(), cont, landingPad,
          callOp.getOperands());
      tryCall.setExtraAttrsAttr(extraAttrs);
      if (ast)
        tryCall.setAstAttr(*ast);
    }

    // Quick block cleanup: no indirection to the post try block.
    auto brOp = dyn_cast<mlir::cir::BrOp>(afterTry->getTerminator());
    mlir::Block *srcBlock = brOp.getDest();
    rewriter.eraseOp(brOp);
    rewriter.mergeBlocks(srcBlock, afterTry);
    return mlir::success();
  }
};

class CIRLoopOpInterfaceFlattening
    : public mlir::OpInterfaceRewritePattern<mlir::cir::LoopOpInterface> {
public:
  using mlir::OpInterfaceRewritePattern<
      mlir::cir::LoopOpInterface>::OpInterfaceRewritePattern;

  inline void lowerConditionOp(mlir::cir::ConditionOp op, mlir::Block *body,
                               mlir::Block *exit,
                               mlir::PatternRewriter &rewriter) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<mlir::cir::BrCondOp>(op, op.getCondition(),
                                                     body, exit);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoopOpInterface op,
                  mlir::PatternRewriter &rewriter) const final {
    // Setup CFG blocks.
    auto *entry = rewriter.getInsertionBlock();
    auto *exit = rewriter.splitBlock(entry, rewriter.getInsertionPoint());
    auto *cond = &op.getCond().front();
    auto *body = &op.getBody().front();
    auto *step = (op.maybeGetStep() ? &op.maybeGetStep()->front() : nullptr);

    // Setup loop entry branch.
    rewriter.setInsertionPointToEnd(entry);
    rewriter.create<mlir::cir::BrOp>(op.getLoc(), &op.getEntry().front());

    // Branch from condition region to body or exit.
    auto conditionOp = cast<mlir::cir::ConditionOp>(cond->getTerminator());
    lowerConditionOp(conditionOp, body, exit, rewriter);

    // TODO(cir): Remove the walks below. It visits operations unnecessarily,
    // however, to solve this we would likely need a custom DialecConversion
    // driver to customize the order that operations are visited.

    // Lower continue statements.
    mlir::Block *dest = (step ? step : cond);
    op.walkBodySkippingNestedLoops([&](mlir::Operation *op) {
      if (isa<mlir::cir::ContinueOp>(op))
        lowerTerminator(op, dest, rewriter);
    });

    // Lower break statements.
    walkRegionSkipping<mlir::cir::LoopOpInterface, mlir::cir::SwitchOp>(
        op.getBody(), [&](mlir::Operation *op) {
          if (isa<mlir::cir::BreakOp>(op))
            lowerTerminator(op, exit, rewriter);
        });

    // Lower optional body region yield.
    for (auto &blk : op.getBody().getBlocks()) {
      auto bodyYield = dyn_cast<mlir::cir::YieldOp>(blk.getTerminator());
      if (bodyYield)
        lowerTerminator(bodyYield, (step ? step : cond), rewriter);
    }

    // Lower mandatory step region yield.
    if (step)
      lowerTerminator(cast<mlir::cir::YieldOp>(step->getTerminator()), cond,
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

class CIRSwitchOpFlattening
    : public mlir::OpRewritePattern<mlir::cir::SwitchOp> {
public:
  using OpRewritePattern<mlir::cir::SwitchOp>::OpRewritePattern;

  inline void rewriteYieldOp(mlir::PatternRewriter &rewriter,
                             mlir::cir::YieldOp yieldOp,
                             mlir::Block *destination) const {
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(yieldOp, yieldOp.getOperands(),
                                                 destination);
  }

  // Return the new defaultDestination block.
  Block *condBrToRangeDestination(mlir::cir::SwitchOp op,
                                  mlir::PatternRewriter &rewriter,
                                  mlir::Block *rangeDestination,
                                  mlir::Block *defaultDestination,
                                  APInt lowerBound, APInt upperBound) const {
    assert(lowerBound.sle(upperBound) && "Invalid range");
    auto resBlock = rewriter.createBlock(defaultDestination);
    auto sIntType = mlir::cir::IntType::get(op.getContext(), 32, true);
    auto uIntType = mlir::cir::IntType::get(op.getContext(), 32, false);

    auto rangeLength = rewriter.create<mlir::cir::ConstantOp>(
        op.getLoc(), sIntType,
        mlir::cir::IntAttr::get(op.getContext(), sIntType,
                                upperBound - lowerBound));

    auto lowerBoundValue = rewriter.create<mlir::cir::ConstantOp>(
        op.getLoc(), sIntType,
        mlir::cir::IntAttr::get(op.getContext(), sIntType, lowerBound));
    auto diffValue = rewriter.create<mlir::cir::BinOp>(
        op.getLoc(), sIntType, mlir::cir::BinOpKind::Sub, op.getCondition(),
        lowerBoundValue);

    // Use unsigned comparison to check if the condition is in the range.
    auto uDiffValue = rewriter.create<mlir::cir::CastOp>(
        op.getLoc(), uIntType, CastKind::integral, diffValue);
    auto uRangeLength = rewriter.create<mlir::cir::CastOp>(
        op.getLoc(), uIntType, CastKind::integral, rangeLength);

    auto cmpResult = rewriter.create<mlir::cir::CmpOp>(
        op.getLoc(), mlir::cir::BoolType::get(op.getContext()),
        mlir::cir::CmpOpKind::le, uDiffValue, uRangeLength);
    rewriter.create<mlir::cir::BrCondOp>(op.getLoc(), cmpResult,
                                         rangeDestination, defaultDestination);
    return resBlock;
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::SwitchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Empty switch statement: just erase it.
    if (!op.getCases().has_value() || op.getCases()->empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // Create exit block.
    rewriter.setInsertionPointAfter(op);
    auto *exitBlock =
        rewriter.splitBlock(rewriter.getBlock(), rewriter.getInsertionPoint());

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

    // Track fallthrough between cases.
    mlir::cir::YieldOp fallthroughYieldOp = nullptr;

    // Digest the case statements values and bodies.
    for (size_t i = 0; i < op.getCases()->size(); ++i) {
      auto &region = op.getRegion(i);
      auto caseAttr = cast<mlir::cir::CaseAttr>(op.getCases()->getValue()[i]);

      // Found default case: save destination and operands.
      switch (caseAttr.getKind().getValue()) {
      case mlir::cir::CaseOpKind::Default:
        defaultDestination = &region.front();
        defaultOperands = region.getArguments();
        break;
      case mlir::cir::CaseOpKind::Range:
        assert(caseAttr.getValue().size() == 2 &&
               "Case range should have 2 case value");
        rangeValues.push_back(
            {cast<mlir::cir::IntAttr>(caseAttr.getValue()[0]).getValue(),
             cast<mlir::cir::IntAttr>(caseAttr.getValue()[1]).getValue()});
        rangeDestinations.push_back(&region.front());
        rangeOperands.push_back(region.getArguments());
        break;
      case mlir::cir::CaseOpKind::Anyof:
      case mlir::cir::CaseOpKind::Equal:
        // AnyOf cases kind can have multiple values, hence the loop below.
        for (auto &value : caseAttr.getValue()) {
          caseValues.push_back(cast<mlir::cir::IntAttr>(value).getValue());
          caseOperands.push_back(region.getArguments());
          caseDestinations.push_back(&region.front());
        }
        break;
      }

      // Previous case is a fallthrough: branch it to this case.
      if (fallthroughYieldOp) {
        rewriteYieldOp(rewriter, fallthroughYieldOp, &region.front());
        fallthroughYieldOp = nullptr;
      }

      for (auto &blk : region.getBlocks()) {
        if (blk.getNumSuccessors())
          continue;

        // Handle switch-case yields.
        if (auto yieldOp = dyn_cast<mlir::cir::YieldOp>(blk.getTerminator()))
          fallthroughYieldOp = yieldOp;
      }

      // Handle break statements.
      walkRegionSkipping<mlir::cir::LoopOpInterface, mlir::cir::SwitchOp>(
          region, [&](mlir::Operation *op) {
            if (isa<mlir::cir::BreakOp>(op))
              lowerTerminator(op, exitBlock, rewriter);
          });

      // Extract region contents before erasing the switch op.
      rewriter.inlineRegionBefore(region, exitBlock);
    }

    // Last case is a fallthrough: branch it to exit.
    if (fallthroughYieldOp) {
      rewriteYieldOp(rewriter, fallthroughYieldOp, exitBlock);
      fallthroughYieldOp = nullptr;
    }

    for (size_t index = 0; index < rangeValues.size(); ++index) {
      auto lowerBound = rangeValues[index].first;
      auto upperBound = rangeValues[index].second;

      // The case range is unreachable, skip it.
      if (lowerBound.sgt(upperBound))
        continue;

      // If range is small, add multiple switch instruction cases.
      // This magical number is from the original CGStmt code.
      constexpr int kSmallRangeThreshold = 64;
      if ((upperBound - lowerBound)
              .ult(llvm::APInt(32, kSmallRangeThreshold))) {
        for (auto iValue = lowerBound; iValue.sle(upperBound); (void)iValue++) {
          caseValues.push_back(iValue);
          caseOperands.push_back(rangeOperands[index]);
          caseDestinations.push_back(rangeDestinations[index]);
        }
        continue;
      }

      defaultDestination =
          condBrToRangeDestination(op, rewriter, rangeDestinations[index],
                                   defaultDestination, lowerBound, upperBound);
      defaultOperands = rangeOperands[index];
    }

    // Set switch op to branch to the newly created blocks.
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<mlir::cir::SwitchFlatOp>(
        op, op.getCondition(), defaultDestination, defaultOperands, caseValues,
        caseDestinations, caseOperands);

    return mlir::success();
  }
};
class CIRTernaryOpFlattening
    : public mlir::OpRewritePattern<mlir::cir::TernaryOp> {
public:
  using OpRewritePattern<mlir::cir::TernaryOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::TernaryOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *condBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
    SmallVector<mlir::Location, 2> locs;
    // Ternary result is optional, make sure to populate the location only
    // when relevant.
    if (op->getResultTypes().size())
      locs.push_back(loc);
    auto *continueBlock =
        rewriter.createBlock(remainingOpsBlock, op->getResultTypes(), locs);
    rewriter.create<mlir::cir::BrOp>(loc, remainingOpsBlock);

    auto &trueRegion = op.getTrueRegion();
    auto *trueBlock = &trueRegion.front();
    mlir::Operation *trueTerminator = trueRegion.back().getTerminator();
    rewriter.setInsertionPointToEnd(&trueRegion.back());
    auto trueYieldOp = dyn_cast<mlir::cir::YieldOp>(trueTerminator);

    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
        trueYieldOp, trueYieldOp.getArgs(), continueBlock);
    rewriter.inlineRegionBefore(trueRegion, continueBlock);

    auto *falseBlock = continueBlock;
    auto &falseRegion = op.getFalseRegion();

    falseBlock = &falseRegion.front();
    mlir::Operation *falseTerminator = falseRegion.back().getTerminator();
    rewriter.setInsertionPointToEnd(&falseRegion.back());
    auto falseYieldOp = dyn_cast<mlir::cir::YieldOp>(falseTerminator);
    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
        falseYieldOp, falseYieldOp.getArgs(), continueBlock);
    rewriter.inlineRegionBefore(falseRegion, continueBlock);

    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<mlir::cir::BrCondOp>(loc, op.getCond(), trueBlock,
                                         falseBlock);

    rewriter.replaceOp(op, continueBlock->getArguments());

    // Ok, we're done!
    return mlir::success();
  }
};

void populateFlattenCFGPatterns(RewritePatternSet &patterns) {
  patterns
      .add<CIRIfFlattening, CIRLoopOpInterfaceFlattening, CIRScopeOpFlattening,
           CIRSwitchOpFlattening, CIRTernaryOpFlattening, CIRTryOpFlattening>(
          patterns.getContext());
}

void FlattenCFGPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateFlattenCFGPatterns(patterns);

  // Collect operations to apply patterns.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
    if (isa<IfOp, ScopeOp, SwitchOp, LoopOpInterface, TernaryOp, TryOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (applyOpPatternsAndFold(ops, std::move(patterns)).failed())
    signalPassFailure();
}

} // namespace

namespace mlir {

std::unique_ptr<Pass> createFlattenCFGPass() {
  return std::make_unique<FlattenCFGPass>();
}

} // namespace mlir

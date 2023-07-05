//===- MergeCleanups.cpp - merge simple return/yield blocks ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace cir;

namespace {

template <typename ScopeLikeOpTy>
struct SimplifyRetYieldBlocks : public mlir::OpRewritePattern<ScopeLikeOpTy> {
  using OpRewritePattern<ScopeLikeOpTy>::OpRewritePattern;
  mlir::LogicalResult replaceScopeLikeOp(PatternRewriter &rewriter,
                                         ScopeLikeOpTy scopeLikeOp) const;

  SimplifyRetYieldBlocks(mlir::MLIRContext *context)
      : OpRewritePattern<ScopeLikeOpTy>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  checkAndRewriteRegion(mlir::Region &r,
                        mlir::PatternRewriter &rewriter) const {
    auto &blocks = r.getBlocks();

    if (blocks.size() <= 1)
      return failure();

    // Rewrite something like this:
    //
    // cir.if %2 {
    //    %3 = cir.const(3 : i32) : i32
    //    cir.br ^bb1
    //  ^bb1:  // pred: ^bb0
    //    cir.return %3 : i32
    //  }
    //
    // to this:
    //
    // cir.if %2 {
    //    %3 = cir.const(3 : i32) : i32
    //    cir.return %3 : i32
    // }
    //
    SmallPtrSet<mlir::Block *, 4> candidateBlocks;
    for (Block &block : blocks) {
      if (block.isEntryBlock())
        continue;

      auto yieldVars = block.getOps<cir::YieldOp>();
      for (cir::YieldOp yield : yieldVars)
        candidateBlocks.insert(yield.getOperation()->getBlock());

      auto retVars = block.getOps<cir::ReturnOp>();
      for (cir::ReturnOp ret : retVars)
        candidateBlocks.insert(ret.getOperation()->getBlock());
    }

    auto changed = mlir::failure();
    for (auto *mergeSource : candidateBlocks) {
      if (!(mergeSource->hasNoSuccessors() && mergeSource->hasOneUse()))
        continue;
      auto *mergeDest = mergeSource->getSinglePredecessor();
      if (!mergeDest || mergeDest->getNumSuccessors() != 1)
        continue;
      rewriter.eraseOp(mergeDest->getTerminator());
      rewriter.mergeBlocks(mergeSource, mergeDest);
      changed = mlir::success();
    }

    return changed;
  }

  mlir::LogicalResult
  checkAndRewriteLoopCond(mlir::Region &condRegion,
                          mlir::PatternRewriter &rewriter) const {
    SmallVector<Operation *> opsToSimplify;
    condRegion.walk([&](Operation *op) {
      if (isa<cir::BrCondOp>(op))
        opsToSimplify.push_back(op);
    });

    // Blocks should only contain one "yield" operation.
    auto trivialYield = [&](Block *b) {
      if (&b->front() != &b->back())
        return false;
      return isa<YieldOp>(b->getTerminator());
    };

    if (opsToSimplify.size() != 1)
      return failure();
    BrCondOp brCondOp = cast<cir::BrCondOp>(opsToSimplify[0]);

    // TODO: leverage SCCP to get improved results.
    auto cstOp = dyn_cast<cir::ConstantOp>(brCondOp.getCond().getDefiningOp());
    if (!cstOp || !cstOp.getValue().isa<mlir::cir::BoolAttr>() ||
        !trivialYield(brCondOp.getDestTrue()) ||
        !trivialYield(brCondOp.getDestFalse()))
      return failure();

    // If the condition is constant, no need to use brcond, just yield
    // properly, "yield" for false and "yield continue" for true.
    auto boolAttr = cstOp.getValue().cast<mlir::cir::BoolAttr>();
    auto *falseBlock = brCondOp.getDestFalse();
    auto *trueBlock = brCondOp.getDestTrue();
    auto *currBlock = brCondOp.getOperation()->getBlock();
    if (boolAttr.getValue()) {
      rewriter.eraseOp(opsToSimplify[0]);
      rewriter.mergeBlocks(trueBlock, currBlock);
      falseBlock->erase();
    } else {
      rewriter.eraseOp(opsToSimplify[0]);
      rewriter.mergeBlocks(falseBlock, currBlock);
      trueBlock->erase();
    }
    if (cstOp.use_empty())
      rewriter.eraseOp(cstOp);
    return success();
  }

  mlir::LogicalResult
  matchAndRewrite(ScopeLikeOpTy op,
                  mlir::PatternRewriter &rewriter) const override {
    return replaceScopeLikeOp(rewriter, op);
  }
};

// Specialize the template to account for the different build signatures for
// IfOp, ScopeOp, FuncOp, SwitchOp, LoopOp.
template <>
mlir::LogicalResult
SimplifyRetYieldBlocks<IfOp>::replaceScopeLikeOp(PatternRewriter &rewriter,
                                                 IfOp ifOp) const {
  auto regionChanged = mlir::failure();
  if (checkAndRewriteRegion(ifOp.getThenRegion(), rewriter).succeeded())
    regionChanged = mlir::success();
  if (checkAndRewriteRegion(ifOp.getElseRegion(), rewriter).succeeded())
    regionChanged = mlir::success();
  return regionChanged;
}

template <>
mlir::LogicalResult
SimplifyRetYieldBlocks<ScopeOp>::replaceScopeLikeOp(PatternRewriter &rewriter,
                                                    ScopeOp scopeOp) const {
  // Scope region empty: just remove scope.
  if (scopeOp.getRegion().empty()) {
    rewriter.eraseOp(scopeOp);
    return mlir::success();
  }

  // Scope region non-empty: clean it up.
  if (checkAndRewriteRegion(scopeOp.getRegion(), rewriter).succeeded())
    return mlir::success();

  return mlir::failure();
}

template <>
mlir::LogicalResult SimplifyRetYieldBlocks<cir::FuncOp>::replaceScopeLikeOp(
    PatternRewriter &rewriter, cir::FuncOp funcOp) const {
  auto regionChanged = mlir::failure();
  if (checkAndRewriteRegion(funcOp.getRegion(), rewriter).succeeded())
    regionChanged = mlir::success();
  return regionChanged;
}

template <>
mlir::LogicalResult SimplifyRetYieldBlocks<cir::SwitchOp>::replaceScopeLikeOp(
    PatternRewriter &rewriter, cir::SwitchOp switchOp) const {
  auto regionChanged = mlir::failure();
  for (auto &r : switchOp.getRegions()) {
    if (checkAndRewriteRegion(r, rewriter).succeeded())
      regionChanged = mlir::success();
  }
  return regionChanged;
}

template <>
mlir::LogicalResult SimplifyRetYieldBlocks<cir::LoopOp>::replaceScopeLikeOp(
    PatternRewriter &rewriter, cir::LoopOp loopOp) const {
  auto regionChanged = mlir::failure();
  if (checkAndRewriteRegion(loopOp.getBody(), rewriter).succeeded())
    regionChanged = mlir::success();
  if (checkAndRewriteLoopCond(loopOp.getCond(), rewriter).succeeded())
    regionChanged = mlir::success();
  return regionChanged;
}

void getMergeCleanupsPatterns(RewritePatternSet &results,
                              MLIRContext *context) {
  results.add<SimplifyRetYieldBlocks<IfOp>, SimplifyRetYieldBlocks<ScopeOp>,
              SimplifyRetYieldBlocks<cir::FuncOp>,
              SimplifyRetYieldBlocks<cir::SwitchOp>,
              SimplifyRetYieldBlocks<cir::LoopOp>>(context);
}

struct MergeCleanupsPass : public MergeCleanupsBase<MergeCleanupsPass> {
  MergeCleanupsPass() = default;
  void runOnOperation() override;
};

// The same operation rewriting done here could have been performed
// by CanonicalizerPass (adding hasCanonicalizer for target Ops and implementing
// the same from above in CIRDialects.cpp). However, it's currently too
// aggressive for static analysis purposes, since it might remove things where
// a diagnostic can be generated.
//
// FIXME: perhaps we can add one more mode to GreedyRewriteConfig to
// disable this behavior.
void MergeCleanupsPass::runOnOperation() {
  auto op = getOperation();
  mlir::RewritePatternSet patterns(&getContext());
  getMergeCleanupsPatterns(patterns, &getContext());
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  SmallVector<Operation *> opsToSimplify;
  op->walk([&](Operation *op) {
    if (isa<cir::IfOp, cir::ScopeOp, cir::FuncOp, cir::SwitchOp, cir::LoopOp>(
            op))
      opsToSimplify.push_back(op);
  });

  for (auto *o : opsToSimplify) {
    bool erase = false;
    (void)applyOpPatternsAndFold(o, frozenPatterns, GreedyRewriteConfig(),
                                 &erase);
  }
}
} // namespace

std::unique_ptr<Pass> mlir::createMergeCleanupsPass() {
  return std::make_unique<MergeCleanupsPass>();
}

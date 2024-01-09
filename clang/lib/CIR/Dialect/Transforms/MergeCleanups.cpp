//===- MergeCleanups.cpp - merge simple return/yield blocks ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

namespace {

/// Removes branches between two blocks if it is the only branch.
///
/// From:
///   ^bb0:
///     cir.br ^bb1
///   ^bb1:  // pred: ^bb0
///     cir.return
///
/// To:
///   ^bb0:
///     cir.return
struct RemoveRedudantBranches : public OpRewritePattern<BrOp> {
  using OpRewritePattern<BrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BrOp op,
                                PatternRewriter &rewriter) const final {
    Block *block = op.getOperation()->getBlock();
    Block *dest = op.getDest();

    // Single edge between blocks: merge it.
    if (block->getNumSuccessors() == 1 &&
        dest->getSinglePredecessor() == block) {
      rewriter.eraseOp(op);
      rewriter.mergeBlocks(dest, block);
      return success();
    }

    return failure();
  }
};

/// Merges basic blocks of trivial conditional branches. This is useful when a
/// the condition of conditional branch is a constant and the destinations of
/// the conditional branch both have only one predecessor.
///
/// From:
///   ^bb0:
///     %0 = cir.const(#true) : !cir.bool
///     cir.brcond %0 ^bb1, ^bb2
///   ^bb1: // pred: ^bb0
///     cir.yield continue
///   ^bb2: // pred: ^bb0
///     cir.yield
///
/// To:
///   ^bb0:
///     cir.yield continue
///
struct MergeTrivialConditionalBranches : public OpRewritePattern<BrCondOp> {
  using OpRewritePattern<BrCondOp>::OpRewritePattern;

  LogicalResult match(BrCondOp op) const final {
    return success(isa<ConstantOp>(op.getCond().getDefiningOp()) &&
                   op.getDestFalse()->hasOneUse() &&
                   op.getDestTrue()->hasOneUse());
  }

  /// Replace conditional branch with unconditional branch.
  void rewrite(BrCondOp op, PatternRewriter &rewriter) const final {
    auto constOp = llvm::cast<ConstantOp>(op.getCond().getDefiningOp());
    bool cond = constOp.getValue().cast<cir::BoolAttr>().getValue();
    Block *block = op.getOperation()->getBlock();

    rewriter.eraseOp(op);
    if (cond) {
      rewriter.mergeBlocks(op.getDestTrue(), block);
      rewriter.eraseBlock(op.getDestFalse());
    } else {
      rewriter.mergeBlocks(op.getDestFalse(), block);
      rewriter.eraseBlock(op.getDestTrue());
    }
  }
};

struct RemoveEmptyScope : public OpRewritePattern<ScopeOp> {
  using OpRewritePattern<ScopeOp>::OpRewritePattern;

  LogicalResult match(ScopeOp op) const final {
    return success(op.getRegion().empty() ||
                   (op.getRegion().getBlocks().size() == 1 &&
                    op.getRegion().front().empty()));
  }

  void rewrite(ScopeOp op, PatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
  }
};

struct RemoveEmptySwitch : public OpRewritePattern<SwitchOp> {
  using OpRewritePattern<SwitchOp>::OpRewritePattern;

  LogicalResult match(SwitchOp op) const final {
    return success(op.getRegions().empty());
  }

  void rewrite(SwitchOp op, PatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
  }
};

//===----------------------------------------------------------------------===//
// MergeCleanupsPass
//===----------------------------------------------------------------------===//

struct MergeCleanupsPass : public MergeCleanupsBase<MergeCleanupsPass> {
  using MergeCleanupsBase::MergeCleanupsBase;

  // The same operation rewriting done here could have been performed
  // by CanonicalizerPass (adding hasCanonicalizer for target Ops and
  // implementing the same from above in CIRDialects.cpp). However, it's
  // currently too aggressive for static analysis purposes, since it might
  // remove things where a diagnostic can be generated.
  //
  // FIXME: perhaps we can add one more mode to GreedyRewriteConfig to
  // disable this behavior.
  void runOnOperation() override;
};

void populateMergeCleanupPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    RemoveRedudantBranches,
    MergeTrivialConditionalBranches,
    RemoveEmptyScope,
    RemoveEmptySwitch
  >(patterns.getContext());
  // clang-format on
}

void MergeCleanupsPass::runOnOperation() {
  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  populateMergeCleanupPatterns(patterns);

  // Collect operations to apply patterns.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](Operation *op) {
    if (isa<BrOp, BrCondOp, ScopeOp, SwitchOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (applyOpPatternsAndFold(ops, std::move(patterns)).failed())
    signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> mlir::createMergeCleanupsPass() {
  return std::make_unique<MergeCleanupsPass>();
}

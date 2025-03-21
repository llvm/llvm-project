//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements pass that canonicalizes CIR operations, eliminating
// redundant branches, empty scopes, and other unnecessary operations.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/MissingFeatures.h"

using namespace mlir;
using namespace cir;

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
struct RemoveRedundantBranches : public OpRewritePattern<BrOp> {
  using OpRewritePattern<BrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BrOp op,
                                PatternRewriter &rewriter) const final {
    Block *block = op.getOperation()->getBlock();
    Block *dest = op.getDest();

    assert(!cir::MissingFeatures::labelOp());

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

struct RemoveEmptyScope
    : public OpRewritePattern<ScopeOp>::SplitMatchAndRewrite {
  using SplitMatchAndRewrite::SplitMatchAndRewrite;

  LogicalResult match(ScopeOp op) const final {
    // TODO: Remove this logic once CIR uses MLIR infrastructure to remove
    // trivially dead operations
    if (op.isEmpty())
      return success();

    Region &region = op.getScopeRegion();
    if (region.getBlocks().front().getOperations().size() == 1)
      return success(isa<YieldOp>(region.getBlocks().front().front()));

    return failure();
  }

  void rewrite(ScopeOp op, PatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
  }
};

//===----------------------------------------------------------------------===//
// CIRCanonicalizePass
//===----------------------------------------------------------------------===//

struct CIRCanonicalizePass : public CIRCanonicalizeBase<CIRCanonicalizePass> {
  using CIRCanonicalizeBase::CIRCanonicalizeBase;

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

void populateCIRCanonicalizePatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    RemoveRedundantBranches,
    RemoveEmptyScope
  >(patterns.getContext());
  // clang-format on
}

void CIRCanonicalizePass::runOnOperation() {
  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  populateCIRCanonicalizePatterns(patterns);

  // Collect operations to apply patterns.
  llvm::SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](Operation *op) {
    assert(!cir::MissingFeatures::switchOp());
    assert(!cir::MissingFeatures::tryOp());
    assert(!cir::MissingFeatures::selectOp());
    assert(!cir::MissingFeatures::complexCreateOp());
    assert(!cir::MissingFeatures::complexRealOp());
    assert(!cir::MissingFeatures::complexImagOp());
    assert(!cir::MissingFeatures::callOp());
    // CastOp and UnaryOp are here to perform a manual `fold` in
    // applyOpPatternsGreedily.
    if (isa<BrOp, BrCondOp, ScopeOp, CastOp, UnaryOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (applyOpPatternsGreedily(ops, std::move(patterns)).failed())
    signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> mlir::createCIRCanonicalizePass() {
  return std::make_unique<CIRCanonicalizePass>();
}

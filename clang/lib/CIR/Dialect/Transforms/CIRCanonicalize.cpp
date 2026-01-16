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

namespace mlir {
#define GEN_PASS_DEF_CIRCANONICALIZE
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

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

    if (isa<cir::LabelOp, cir::IndirectBrOp>(dest->front()))
      return failure();
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

//===----------------------------------------------------------------------===//
// CIRCanonicalizePass
//===----------------------------------------------------------------------===//

struct CIRCanonicalizePass
    : public impl::CIRCanonicalizeBase<CIRCanonicalizePass> {
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
    RemoveRedundantBranches
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
    assert(!cir::MissingFeatures::tryOp());
    assert(!cir::MissingFeatures::callOp());

    // Many operations are here to perform a manual `fold` in
    // applyOpPatternsGreedily.
    if (isa<BrOp, BrCondOp, CastOp, ScopeOp, SwitchOp, SelectOp, UnaryOp,
            ComplexCreateOp, ComplexImagOp, ComplexRealOp, VecCmpOp,
            VecCreateOp, VecExtractOp, VecShuffleOp, VecShuffleDynamicOp,
            VecTernaryOp, BitClrsbOp, BitClzOp, BitCtzOp, BitFfsOp, BitParityOp,
            BitPopcountOp, BitReverseOp, ByteSwapOp, RotateOp, ConstantOp>(op))
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

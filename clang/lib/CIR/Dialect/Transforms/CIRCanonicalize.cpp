//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR canonicalization pass. It applies the
// canonicalization patterns and folders of the CIR dialect to every operation
// until a fixpoint is reached.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_CIRCANONICALIZE
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct CIRCanonicalizePass
    : public impl::CIRCanonicalizeBase<CIRCanonicalizePass> {
  using CIRCanonicalizeBase::CIRCanonicalizeBase;

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    for (RegisteredOperationName op : context->getRegisteredOperationsByDialect(
             cir::CIRDialect::getDialectNamespace()))
      op.getCanonicalizationPatterns(owningPatterns, context);
    patterns =
        std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    return success();
  }

  void runOnOperation() override {
    Region &body = getOperation()->getRegion(0);
    bool changed = applyPatterns(body);
    // A single run must reach a fixpoint, so a second run changes nothing.
    if (testConvergence && applyPatterns(body))
      return signalPassFailure();
    if (!changed)
      markAllAnalysesPreserved();
  }

  /// Applies the patterns to every operation in `body` and returns whether
  /// the IR changed.
  bool applyPatterns(Region &body) {
    // The driver pops from the back of the worklist, so seeding it in reverse
    // pre-order visits the IR top-down.
    SmallVector<Operation *> ops;
    body.walk<WalkOrder::PostOrder, ReverseIterator>(
        [&](Operation *op) { ops.push_back(op); });

    // Use the multi-op driver rather than applyPatternsGreedily on the
    // regions: the region driver always erases blocks that have no CFG
    // predecessor, but before the goto solver runs, a block that starts with
    // a `cir.label` is reached only through `cir.goto`, which is not a CFG
    // edge. For the same reason region simplification is not run.
    //
    // Dead operations are kept: CIR stays close to the source, so an unused
    // local or an ignored expression is still emitted, as in classic codegen.
    GreedyRewriteConfig config;
    config.setScope(&body);
    config.enableDeadCodeElimination(false);
    bool changed = false;
    (void)applyOpPatternsGreedily(ops, *patterns, config, &changed);
    return changed;
  }

  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};

} // namespace

std::unique_ptr<Pass> mlir::createCIRCanonicalizePass() {
  return std::make_unique<CIRCanonicalizePass>();
}

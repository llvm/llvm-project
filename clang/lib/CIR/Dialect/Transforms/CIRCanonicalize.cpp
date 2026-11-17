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
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/CIRTransformUtils.h"

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
    Region &body = getOperation().getBodyRegion();
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

    // CIR's driver also rewrites the code that control enters only through
    // cir.goto or cir.switch, which MLIR's drivers skip as unreachable until
    // cir-flatten-cfg and cir-goto-solver turn those jumps into branches.
    //
    // Only operations that rewriting left dead are erased: CIR stays close to
    // the source, so an unused local or an ignored expression is still
    // emitted, as in classic codegen.
    return cir::applyOpPatternsGreedily(ops, *patterns, body,
                                        /*eraseOnlyNewlyDead=*/true);
  }

  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};

} // namespace

std::unique_ptr<Pass> mlir::createCIRCanonicalizePass() {
  return std::make_unique<CIRCanonicalizePass>();
}

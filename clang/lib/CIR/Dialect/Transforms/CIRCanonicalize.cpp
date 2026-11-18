//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR canonicalization pass, which applies the
// canonicalization patterns and folders of the CIR dialect.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Config/mlir-config.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/Analysis/GotoAnalysis.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_CIRCANONICALIZE
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

/// Returns true if `op` is a scope, cleanup scope or switch that holds no code:
/// each of its regions is a single block that ends in `cir.yield` and holds
/// only such operations, like `cir.scope {}`. An empty `cir.case` does not
/// count, since it still decides where its switch goes.
static bool holdsNoCode(Operation *op) {
  if (!isa<cir::ScopeOp, cir::CleanupScopeOp, cir::SwitchOp>(op))
    return false;
  return llvm::all_of(op->getRegions(), [](Region &region) {
    if (!region.hasOneBlock())
      return false;
    Block &block = region.front();
    return isa<cir::YieldOp>(block.getTerminator()) &&
           llvm::all_of(block.without_terminator(),
                        [](Operation &nested) { return holdsNoCode(&nested); });
  });
}

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
    bool changed = applyPatterns();
    // A test can check that one run reaches a fixpoint on its input. One run
    // need not in general: the driver does not revisit the parent of an
    // erased operation, which may then fold or hold no code.
    if (testConvergence && applyPatterns())
      return signalPassFailure();
    if (!changed)
      markAllAnalysesPreserved();
  }

  /// Applies the patterns to each region of the pass's operation and returns
  /// whether the IR changed.
  bool applyPatterns() {
    Operation *root = getOperation();
    // Until cir-flatten-cfg and cir-goto-solver turn cir.goto and cir.switch
    // into branches, the code they enter has no CFG predecessor, so
    // GotoAnalysis tells the driver what control reaches. As the driver's
    // listener, it stays up to date while the patterns rewrite the IR.
    cir::GotoAnalysis reachability(root);
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps)
        .setListener(&reachability)
        .setReachability([&](Operation *op) {
          bool reachable = reachability.isReachable(op);
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
          // The analysis may over-approximate after a jump target moves, but
          // it must reach whatever a fresh analysis reaches.
          assert((reachable || !cir::GotoAnalysis(root).isReachable(op)) &&
                 "cached reachability misses an operation that control "
                 "reaches");
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
          return reachable;
        });

    // As MLIR's canonicalizer does, rewrite each region on its own, with the
    // region as the driver's scope.
    bool changed = false;
    for (Region &region : root->getRegions()) {
      // The driver pops from the back of the worklist, so seeding it in
      // reverse pre-order visits the region top-down.
      //
      // An operation that is trivially dead in the input stays, as in classic
      // codegen: it is not seeded, and ExistingAndNewOps keeps the driver away
      // from it unless a rewrite creates or moves it. An operation that holds
      // no code has nothing to keep, so it is seeded and erased. The walk
      // checks nested operations before their parent, which lets the cache
      // reuse what it learned about them.
      SmallVector<Operation *> ops;
      DeadnessCache deadness;
      auto seed = [&](Operation *op) {
        if (!deadness.isOpTriviallyDead(op) || holdsNoCode(op))
          ops.push_back(op);
      };
      for (Block &block : llvm::reverse(region))
        for (Operation &op : llvm::reverse(block))
          op.walk<WalkOrder::PostOrder, ReverseIterator>(seed);

      config.setScope(&region);
      // Without a rewrite limit, the driver empties its worklist and so
      // cannot fail.
      bool regionChanged = false;
      (void)applyOpPatternsGreedily(ops, *patterns, config, &regionChanged);
      changed |= regionChanged;
    }
    return changed;
  }

  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};

} // namespace

std::unique_ptr<Pass> mlir::createCIRCanonicalizePass() {
  return std::make_unique<CIRCanonicalizePass>();
}

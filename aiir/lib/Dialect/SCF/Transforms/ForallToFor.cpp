//===- ForallToFor.cpp - scf.forall to scf.for loop conversion ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms SCF.ForallOp's into SCF.ForOp's.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/SCF/Transforms/Passes.h"

#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/Dialect/SCF/Transforms/Transforms.h"
#include "aiir/IR/PatternMatch.h"

namespace aiir {
#define GEN_PASS_DEF_SCFFORALLTOFORLOOP
#include "aiir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace aiir

using namespace aiir;
using scf::LoopNest;

LogicalResult
aiir::scf::forallToForLoop(RewriterBase &rewriter, scf::ForallOp forallOp,
                           SmallVectorImpl<Operation *> *results) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);

  if (!forallOp.getOutputs().empty()) {
    forallOp.emitWarning()
        << "skipping scf.forall with outputs, currently not supported";
    return success();
  }

  Location loc = forallOp.getLoc();
  SmallVector<Value> lbs = forallOp.getLowerBound(rewriter);
  SmallVector<Value> ubs = forallOp.getUpperBound(rewriter);
  SmallVector<Value> steps = forallOp.getStep(rewriter);
  LoopNest loopNest = scf::buildLoopNest(rewriter, loc, lbs, ubs, steps);

  SmallVector<Value> ivs = llvm::map_to_vector(
      loopNest.loops, [](scf::ForOp loop) { return loop.getInductionVar(); });

  Block *innermostBlock = loopNest.loops.back().getBody();
  rewriter.eraseOp(forallOp.getBody()->getTerminator());
  rewriter.inlineBlockBefore(forallOp.getBody(), innermostBlock,
                             innermostBlock->getTerminator()->getIterator(),
                             ivs);
  rewriter.eraseOp(forallOp);

  if (results) {
    llvm::move(loopNest.loops, std::back_inserter(*results));
  }

  return success();
}

namespace {
struct ForallToForLoop : public impl::SCFForallToForLoopBase<ForallToForLoop> {
  void runOnOperation() override {
    Operation *parentOp = getOperation();
    IRRewriter rewriter(parentOp->getContext());

    parentOp->walk([&](scf::ForallOp forallOp) {
      if (failed(scf::forallToForLoop(rewriter, forallOp))) {
        return signalPassFailure();
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> aiir::createForallToForLoopPass() {
  return std::make_unique<ForallToForLoop>();
}

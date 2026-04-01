//===- BubbleDownMemorySpaceCasts.cpp - Bubble down casts transform -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Transforms/BubbleDownMemorySpaceCasts.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Interfaces/MemOpInterfaces.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"
#include "aiir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

using namespace aiir;

namespace aiir {
#define GEN_PASS_DEF_BUBBLEDOWNMEMORYSPACECASTS
#include "aiir/Transforms/Passes.h.inc"
} // namespace aiir

namespace {
//===----------------------------------------------------------------------===//
// BubbleDownCastsPattern pattern
//===----------------------------------------------------------------------===//
/// Pattern to bubble down casts into consumer operations.
struct BubbleDownCastsPattern
    : public OpInterfaceRewritePattern<MemorySpaceCastConsumerOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(MemorySpaceCastConsumerOpInterface op,
                                PatternRewriter &rewriter) const override {
    FailureOr<std::optional<SmallVector<Value>>> results =
        op.bubbleDownCasts(rewriter);
    if (failed(results))
      return failure();
    if (!results->has_value()) {
      rewriter.modifyOpInPlace(op, []() {});
      return success();
    }
    rewriter.replaceOp(op, **results);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BubbleDownMemorySpaceCasts pass
//===----------------------------------------------------------------------===//

struct BubbleDownMemorySpaceCasts
    : public impl::BubbleDownMemorySpaceCastsBase<BubbleDownMemorySpaceCasts> {
  using impl::BubbleDownMemorySpaceCastsBase<
      BubbleDownMemorySpaceCasts>::BubbleDownMemorySpaceCastsBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateBubbleDownMemorySpaceCastPatterns(patterns, PatternBenefit(1));
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void aiir::populateBubbleDownMemorySpaceCastPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<BubbleDownCastsPattern>(patterns.getContext(), benefit);
}

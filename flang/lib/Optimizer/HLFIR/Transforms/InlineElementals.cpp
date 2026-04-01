//===- InlineElementals.cpp - Inline chained hlfir.elemental ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Chained elemental operations like a + b + c can inline the first elemental
// at the hlfir.apply in the body of the second one (as described in
// docs/HighLevelFIR.md). This has to be done in a pass rather than in lowering
// so that it happens after the HLFIR intrinsic simplification pass.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/IR/IRMapping.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Support/LLVM.h"
#include "aiir/Transforms/DialectConversion.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iterator>

namespace hlfir {
#define GEN_PASS_DEF_INLINEELEMENTALS
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

/// If the elemental has only two uses and those two are an apply operation and
/// a destroy operation, return those two, otherwise return {}
static std::optional<std::pair<hlfir::ApplyOp, hlfir::DestroyOp>>
getTwoUses(hlfir::ElementalOp elemental) {
  aiir::Operation::user_range users = elemental->getUsers();
  // don't inline anything with more than one use (plus hfir.destroy)
  if (std::distance(users.begin(), users.end()) != 2) {
    return std::nullopt;
  }

  // If the ElementalOp must produce a temporary (e.g. for
  // finalization purposes), then we cannot inline it.
  if (hlfir::elementalOpMustProduceTemp(elemental))
    return std::nullopt;

  hlfir::ApplyOp apply;
  hlfir::DestroyOp destroy;
  for (aiir::Operation *user : users)
    aiir::TypeSwitch<aiir::Operation *, void>(user)
        .Case([&](hlfir::ApplyOp op) { apply = op; })
        .Case([&](hlfir::DestroyOp op) { destroy = op; });

  if (!apply || !destroy)
    return std::nullopt;

  // we can't inline if the return type of the yield doesn't match the return
  // type of the apply
  auto yield = aiir::dyn_cast_or_null<hlfir::YieldElementOp>(
      elemental.getRegion().back().back());
  assert(yield && "hlfir.elemental should always end with a yield");
  if (apply.getResult().getType() != yield.getElementValue().getType())
    return std::nullopt;

  return std::pair{apply, destroy};
}

namespace {
class InlineElementalConversion
    : public aiir::OpRewritePattern<hlfir::ElementalOp> {
public:
  using aiir::OpRewritePattern<hlfir::ElementalOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::ElementalOp elemental,
                  aiir::PatternRewriter &rewriter) const override {
    std::optional<std::pair<hlfir::ApplyOp, hlfir::DestroyOp>> maybeTuple =
        getTwoUses(elemental);
    if (!maybeTuple)
      return rewriter.notifyMatchFailure(
          elemental, "hlfir.elemental does not have two uses");

    if (elemental.isOrdered()) {
      // We can only inline the ordered elemental into a loop-like
      // construct that processes the indices in-order and does not
      // have the side effects itself. Adhere to conservative behavior
      // for the time being.
      return rewriter.notifyMatchFailure(elemental,
                                         "hlfir.elemental is ordered");
    }
    auto [apply, destroy] = *maybeTuple;

    assert(elemental.getRegion().hasOneBlock() &&
           "expect elemental region to have one block");

    fir::FirOpBuilder builder{rewriter, elemental.getOperation()};
    builder.setInsertionPointAfter(apply);
    hlfir::YieldElementOp yield = hlfir::inlineElementalOp(
        elemental.getLoc(), builder, elemental, apply.getIndices());

    // remove the old elemental and all of the bookkeeping
    rewriter.replaceOp(apply, {yield.getElementValue()});
    rewriter.eraseOp(yield);
    rewriter.eraseOp(destroy);
    rewriter.eraseOp(elemental);

    return aiir::success();
  }
};

class InlineElementalsPass
    : public hlfir::impl::InlineElementalsBase<InlineElementalsPass> {
public:
  void runOnOperation() override {
    aiir::AIIRContext *context = &getContext();

    aiir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks.
    config.setRegionSimplificationLevel(
        aiir::GreedySimplifyRegionLevel::Disabled);

    aiir::RewritePatternSet patterns(context);
    patterns.insert<InlineElementalConversion>(context);

    if (aiir::failed(aiir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      aiir::emitError(getOperation()->getLoc(),
                      "failure in HLFIR elemental inlining");
      signalPassFailure();
    }
  }
};
} // namespace

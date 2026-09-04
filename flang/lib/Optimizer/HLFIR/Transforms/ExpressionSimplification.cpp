//===- ExpressionSimplification.cpp - Simplify HLFIR expressions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace hlfir {
#define GEN_PASS_DEF_EXPRESSIONSIMPLIFICATION
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

// Get the first user of `op`.
// Note that we consider the first user to be the one on the lowest line of
// the emitted HLFIR. The user iterator considers the opposite.
template <typename UserOp>
static UserOp getFirstUser(mlir::Operation *op) {
  auto it = op->user_begin(), end = op->user_end(), prev = it;
  for (; it != end; prev = it++)
    ;
  if (prev != end)
    if (auto userOp = mlir::dyn_cast<UserOp>(*prev))
      return userOp;
  return {};
}

// Get the last user of `op`.
// Note that we consider the last user to be the one on the highest line of
// the emitted HLFIR. The user iterator considers the opposite.
template <typename UserOp>
static UserOp getLastUser(mlir::Operation *op) {
  if (!op->getUsers().empty())
    if (auto userOp = mlir::dyn_cast<UserOp>(*op->user_begin()))
      return userOp;
  return {};
}

namespace {

// Trim operations can be erased in certain expressions, such as character
// comparisons.
// Since a character comparison appends spaces to the shorter character,
// calls to trim() that are used only in the comparison can be eliminated.
//
// Example:
// `trim(x) == trim(y)`
// can be simplified to
// `x == y`
class EraseTrim : public mlir::OpRewritePattern<hlfir::CharTrimOp> {
public:
  using mlir::OpRewritePattern<hlfir::CharTrimOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::CharTrimOp trimOp,
                  mlir::PatternRewriter &rewriter) const override {
    int trimUses = std::distance(trimOp->use_begin(), trimOp->use_end());
    auto cmpCharOp = getFirstUser<hlfir::CmpCharOp>(trimOp);
    auto destroyOp = getLastUser<hlfir::DestroyOp>(trimOp);
    if (!cmpCharOp || !destroyOp || trimUses != 2)
      return rewriter.notifyMatchFailure(
          trimOp, "hlfir.char_trim is not used (only) by hlfir.cmpchar");

    rewriter.eraseOp(destroyOp);
    rewriter.replaceOp(trimOp, trimOp.getChr());
    return mlir::success();
  }
};

class ExpressionSimplificationPass
    : public hlfir::impl::ExpressionSimplificationBase<
          ExpressionSimplificationPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();

    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks.
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);

    mlir::RewritePatternSet patterns(context);
    patterns.insert<EraseTrim>(context);

    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "failure in HLFIR expression simplification");
      signalPassFailure();
    }
  }
};

} // namespace

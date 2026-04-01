//===- TosaLayerwiseConstantFoldPass.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements constant folding transformations on TOSA operations
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Tosa/Transforms/Passes.h"

#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace aiir {
namespace tosa {
#define GEN_PASS_DEF_TOSALAYERWISECONSTANTFOLDPASS
#include "aiir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace aiir

using namespace aiir;
using namespace aiir::tosa;

namespace {

template <typename... Args>
void addOpsCanonicalizations(AIIRContext *ctx, RewritePatternSet &patterns) {
  (Args::getCanonicalizationPatterns(patterns, ctx), ...);
}

void populateTosaOpsCanonicalizationPatterns(AIIRContext *ctx,
                                             RewritePatternSet &patterns) {
  addOpsCanonicalizations<
#define GET_OP_LIST
#include "aiir/Dialect/Tosa/IR/TosaOps.cpp.inc"
      >(ctx, patterns);
}

struct TosaLayerwiseConstantFoldPass
    : public tosa::impl::TosaLayerwiseConstantFoldPassBase<
          TosaLayerwiseConstantFoldPass> {
  using Base::Base;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    auto func = getOperation();

    aiir::tosa::populateTosaFoldConstantReciprocalPatterns(ctx, patterns);
    aiir::tosa::populateTosaFoldConstantTransposePatterns(ctx, patterns);
    aiir::tosa::populateTosaConstantReduction(ctx, patterns,
                                              aggressiveReduceConstant);
    populateTosaOpsCanonicalizationPatterns(ctx, patterns);

    if (applyPatternsGreedily(func, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace

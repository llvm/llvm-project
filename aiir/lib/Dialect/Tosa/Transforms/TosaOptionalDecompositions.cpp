//===- TosaOptionalDecompositions.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass to apply the Tosa operations decompositions
// exposed as populate functions in
// include/aiir/Dialect/Tosa/Transforms/Passes.h
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Tosa/Transforms/Passes.h"

#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace aiir {
namespace tosa {
#define GEN_PASS_DEF_TOSAOPTIONALDECOMPOSITIONSPASS
#include "aiir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace aiir

using namespace aiir;

namespace {

struct TosaOptionalDecompositions
    : public tosa::impl::TosaOptionalDecompositionsPassBase<
          TosaOptionalDecompositions> {
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    auto func = getOperation();

    aiir::tosa::populateTosaDecomposeTransposeConv(ctx, patterns);
    aiir::tosa::populateTosaDecomposeDepthwise(ctx, patterns);

    if (applyPatternsGreedily(func, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace

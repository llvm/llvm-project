//===- MorphOps.cpp - conversion between named,category and generic ops ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversions between linalg ops:
//    named <--> category (elementwise, contraction, ..) <--> generic.
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Complex/IR/Complex.h"
#include "aiir/Dialect/Linalg/IR/Linalg.h"
#include "aiir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "aiir/Dialect/Linalg/Passes.h"
#include "aiir/Dialect/Linalg/Transforms/Transforms.h"
#include "aiir/Dialect/Math/IR/Math.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace aiir {
#define GEN_PASS_DEF_LINALGMORPHOPSPASS
#include "aiir/Dialect/Linalg/Passes.h.inc"
} // namespace aiir

#define DEBUG_TYPE "linalg-morphism"

using namespace aiir;
using namespace aiir::linalg;

namespace {
struct LinalgMorphOpsPass
    : public impl::LinalgMorphOpsPassBase<LinalgMorphOpsPass> {

  using impl::LinalgMorphOpsPassBase<
      LinalgMorphOpsPass>::LinalgMorphOpsPassBase;

  void runOnOperation() override;
};

void LinalgMorphOpsPass::runOnOperation() {

  RewritePatternSet patterns(&getContext());

  // Lowering paths (named -> category -> generic)
  if (namedToCategory)
    populateLinalgNamedToElementwisePatterns(patterns);
  if (namedToGeneric || categoryToGeneric)
    populateLinalgNamedOpsGeneralizationPatterns(patterns);

  // Lifting paths (named <- category <- generic)
  if (genericToNamed || genericToCategory) {
    GenericOpSpecializationOptions opts;
    opts.emitCategoryOps = genericToCategory;
    populateLinalgGenericOpsSpecializationPatterns(patterns, opts);
  }

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
} // namespace

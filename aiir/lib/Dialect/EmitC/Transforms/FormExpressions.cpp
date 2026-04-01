//===- FormExpressions.cpp - Form C-style expressions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that forms EmitC operations modeling C operators
// into C-style expressions using the emitc.expression op.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/EmitC/IR/EmitC.h"
#include "aiir/Dialect/EmitC/Transforms/Passes.h"
#include "aiir/Dialect/EmitC/Transforms/Transforms.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace aiir {
namespace emitc {
#define GEN_PASS_DEF_FORMEXPRESSIONSPASS
#include "aiir/Dialect/EmitC/Transforms/Passes.h.inc"
} // namespace emitc
} // namespace aiir

using namespace aiir;
using namespace emitc;

namespace {
struct FormExpressionsPass
    : public emitc::impl::FormExpressionsPassBase<FormExpressionsPass> {
  void runOnOperation() override {
    Operation *rootOp = getOperation();
    AIIRContext *context = rootOp->getContext();

    // Wrap each C operator op with an expression op.
    OpBuilder builder(context);
    auto matchFun = [&](Operation *op) {
      if (isa<emitc::CExpressionInterface>(*op) &&
          !op->getParentOfType<emitc::ExpressionOp>() &&
          op->getNumResults() == 1)
        createExpression(op, builder);
    };
    rootOp->walk(matchFun);

    // Fold expressions where possible.
    RewritePatternSet patterns(context);
    populateExpressionPatterns(patterns);

    if (failed(applyPatternsGreedily(rootOp, std::move(patterns))))
      return signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect>();
  }
};
} // namespace

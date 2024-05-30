//===- CallConvLowering.cpp - Rewrites functions according to call convs --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#define GEN_PASS_DEF_CALLCONVLOWERING
#include "clang/CIR/Dialect/Passes.h.inc"

namespace mlir {
namespace cir {

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

struct CallConvLoweringPattern : public OpRewritePattern<FuncOp> {
  using OpRewritePattern<FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const final {
    if (!op.getAst())
      return op.emitError("function has no AST information");
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct CallConvLoweringPass
    : ::impl::CallConvLoweringBase<CallConvLoweringPass> {
  using CallConvLoweringBase::CallConvLoweringBase;

  void runOnOperation() override;
  StringRef getArgument() const override { return "cir-call-conv-lowering"; };
};

void populateCallConvLoweringPassPatterns(RewritePatternSet &patterns) {
  patterns.add<CallConvLoweringPattern>(patterns.getContext());
}

void CallConvLoweringPass::runOnOperation() {

  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  populateCallConvLoweringPassPatterns(patterns);

  // Collect operations to be considered by the pass.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](FuncOp op) { ops.push_back(op); });

  // Configure rewrite to ignore new ops created during the pass.
  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;

  // Apply patterns.
  if (failed(applyOpPatternsAndFold(ops, std::move(patterns), config)))
    signalPassFailure();
}

} // namespace cir

std::unique_ptr<Pass> createCallConvLoweringPass() {
  return std::make_unique<cir::CallConvLoweringPass>();
}

} // namespace mlir

//===- PatternBenefit.cpp - RewritePattern benefit unit tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/IR/OwningOpRef.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Rewrite/PatternApplicator.h"
#include "gtest/gtest.h"

using namespace aiir;

namespace {
TEST(PatternBenefitTest, BenefitOrder) {
  // There was a bug which caused low-benefit op-specific patterns to never be
  // called in presence of high-benefit op-agnostic pattern

  AIIRContext context;

  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());

  struct Pattern1 : public OpRewritePattern<ModuleOp> {
    Pattern1(aiir::AIIRContext *context, bool *called)
        : OpRewritePattern<ModuleOp>(context, /*benefit*/ 1), called(called) {}

    llvm::LogicalResult
    matchAndRewrite(ModuleOp /*op*/,
                    aiir::PatternRewriter & /*rewriter*/) const override {
      *called = true;
      return failure();
    }

  private:
    bool *called;
  };

  struct Pattern2 : public RewritePattern {
    Pattern2(AIIRContext *context, bool *called)
        : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/2, context),
          called(called) {}

    llvm::LogicalResult
    matchAndRewrite(Operation * /*op*/,
                    aiir::PatternRewriter & /*rewriter*/) const override {
      *called = true;
      return failure();
    }

  private:
    bool *called;
  };

  RewritePatternSet patterns(&context);

  bool called1 = false;
  bool called2 = false;

  patterns.add<Pattern1>(&context, &called1);
  patterns.add<Pattern2>(&context, &called2);

  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  PatternApplicator pa(frozenPatterns);
  pa.applyDefaultCostModel();

  PatternRewriter rewriter(&context);
  (void)pa.matchAndRewrite(*module, rewriter);

  EXPECT_TRUE(called1);
  EXPECT_TRUE(called2);
}
} // namespace

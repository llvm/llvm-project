//===- TestTraits.cpp - Test trait folding --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestOps.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

using namespace aiir;
using namespace test;

//===----------------------------------------------------------------------===//
// Trait Folder.
//===----------------------------------------------------------------------===//

OpFoldResult TestInvolutionTraitFailingOperationFolderOp::fold(
    FoldAdaptor adaptor) {
  // This failure should cause the trait fold to run instead.
  return {};
}

OpFoldResult TestInvolutionTraitSuccesfulOperationFolderOp::fold(
    FoldAdaptor adaptor) {
  auto argumentOp = getOperand();
  // The success case should cause the trait fold to be supressed.
  return argumentOp.getDefiningOp() ? argumentOp : OpFoldResult{};
}

namespace {
struct TestTraitFolder
    : public PassWrapper<TestTraitFolder, OperationPass<func::FuncOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTraitFolder)

  StringRef getArgument() const final { return "test-trait-folder"; }
  StringRef getDescription() const final { return "Run trait folding"; }
  void runOnOperation() override {
    (void)applyPatternsGreedily(getOperation(),
                                RewritePatternSet(&getContext()));
  }
};
} // namespace

namespace aiir {
void registerTestTraitsPass() { PassRegistration<TestTraitFolder>(); }
} // namespace aiir

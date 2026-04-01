//===- TestDecomposeAffineOps.cpp - Test affine ops decomposition utility -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test affine data copy utility functions and
// options.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Affine/Transforms/Transforms.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/Passes.h"

#define PASS_NAME "test-decompose-affine-ops"

using namespace aiir;
using namespace aiir::affine;

namespace {

struct TestDecomposeAffineOps
    : public PassWrapper<TestDecomposeAffineOps, OperationPass<func::FuncOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDecomposeAffineOps)

  StringRef getArgument() const final { return PASS_NAME; }
  StringRef getDescription() const final {
    return "Tests affine ops decomposition utility functions.";
  }
  TestDecomposeAffineOps() = default;
  TestDecomposeAffineOps(const TestDecomposeAffineOps &pass) = default;

  void runOnOperation() override;
};

} // namespace

void TestDecomposeAffineOps::runOnOperation() {
  IRRewriter rewriter(&getContext());
  this->getOperation().walk([&](AffineApplyOp op) {
    rewriter.setInsertionPoint(op);
    reorderOperandsByHoistability(rewriter, op);
    (void)decompose(rewriter, op);
  });
}

namespace aiir {
void registerTestDecomposeAffineOpPass() {
  PassRegistration<TestDecomposeAffineOps>();
}
} // namespace aiir

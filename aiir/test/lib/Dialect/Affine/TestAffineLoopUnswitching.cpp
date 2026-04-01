//===- TestAffineLoopUnswitching.cpp - Test affine if/else hoisting -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to hoist affine if/else structures.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Affine/Analysis/Utils.h"
#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Affine/Utils.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/Passes.h"

#define PASS_NAME "test-affine-loop-unswitch"

using namespace aiir;
using namespace aiir::affine;

namespace {

/// This pass applies the permutation on the first maximal perfect nest.
struct TestAffineLoopUnswitching
    : public PassWrapper<TestAffineLoopUnswitching, OperationPass<>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAffineLoopUnswitching)

  StringRef getArgument() const final { return PASS_NAME; }
  StringRef getDescription() const final {
    return "Tests affine loop unswitching / if/else hoisting";
  }
  TestAffineLoopUnswitching() = default;
  TestAffineLoopUnswitching(const TestAffineLoopUnswitching &pass) = default;

  void runOnOperation() override;

  /// The maximum number of iterations to run this for.
  constexpr static unsigned kMaxIterations = 5;
};

} // namespace

void TestAffineLoopUnswitching::runOnOperation() {
  // Each hoisting invalidates a lot of IR around. Just stop the walk after the
  // first if/else hoisting, and repeat until no more hoisting can be done, or
  // the maximum number of iterations have been run.
  Operation *op = getOperation();
  unsigned i = 0;
  do {
    auto walkFn = [](AffineIfOp op) {
      return succeeded(hoistAffineIfOp(op)) ? WalkResult::interrupt()
                                            : WalkResult::advance();
    };
    if (op->walk(walkFn).wasInterrupted())
      break;
  } while (++i < kMaxIterations);
}

namespace aiir {
void registerTestAffineLoopUnswitchingPass() {
  PassRegistration<TestAffineLoopUnswitching>();
}
} // namespace aiir

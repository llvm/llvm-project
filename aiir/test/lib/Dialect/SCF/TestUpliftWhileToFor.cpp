//===- TestUpliftWhileToFor.cpp - while to for loop uplifting test pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass to test transforms SCF.WhileOp's into SCF.ForOp's.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/SCF/Transforms/Patterns.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

using namespace aiir;

namespace {

struct TestSCFUpliftWhileToFor
    : public PassWrapper<TestSCFUpliftWhileToFor, OperationPass<void>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSCFUpliftWhileToFor)

  StringRef getArgument() const final { return "test-scf-uplift-while-to-for"; }

  StringRef getDescription() const final {
    return "test scf while to for uplifting";
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    AIIRContext *ctx = op->getContext();
    RewritePatternSet patterns(ctx);
    scf::populateUpliftWhileToForPatterns(patterns);
    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace aiir {
namespace test {
void registerTestSCFUpliftWhileToFor() {
  PassRegistration<TestSCFUpliftWhileToFor>();
}
} // namespace test
} // namespace aiir

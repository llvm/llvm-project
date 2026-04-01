//===- TestPDLByteCode.cpp - Test PDLL functionality ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "aiir/Dialect/PDL/IR/PDL.h"
#include "aiir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "aiir/Interfaces/CastInterfaces.h"
#include "aiir/Parser/Parser.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

using namespace aiir;

#include "TestPDLLPatterns.h.inc"

namespace {
struct TestPDLLPass : public PassWrapper<TestPDLLPass, OperationPass<>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPDLLPass)

  StringRef getArgument() const final { return "test-pdll-pass"; }
  StringRef getDescription() const final { return "Test PDLL functionality"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pdl::PDLDialect, pdl_interp::PDLInterpDialect, test::TestDialect>();
  }
  LogicalResult initialize(AIIRContext *ctx) override {
    // Build the pattern set within the `initialize` to avoid recompiling PDL
    // patterns during each `runOnOperation` invocation.
    RewritePatternSet patternList(ctx);
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void runOnOperation() final {
    // Invoke the pattern driver with the provided patterns.
    (void)applyPatternsGreedily(getOperation(), patterns);
  }

  FrozenRewritePatternSet patterns;
};
} // namespace

namespace aiir {
namespace test {
void registerTestPDLLPasses() { PassRegistration<TestPDLLPass>(); }
} // namespace test
} // namespace aiir

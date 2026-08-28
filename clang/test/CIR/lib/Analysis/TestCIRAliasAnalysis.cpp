//===- TestCIRAliasAnalysis.cpp - Test CIR alias analysis results ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/test/lib/Analysis/TestAliasAnalysis.h"
#include "clang/CIR/Dialect/Analysis/CIRAliasAnalysis.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Testing AliasResult
//===----------------------------------------------------------------------===//

struct TestCIRAliasAnalysisPass
    : public test::TestAliasAnalysisBase,
      PassWrapper<TestCIRAliasAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCIRAliasAnalysisPass)

  StringRef getArgument() const final { return "test-cir-alias-analysis"; }
  StringRef getDescription() const final {
    return "Test CIR alias analysis results.";
  }
  void runOnOperation() override {
    mlir::AliasAnalysis aliasAnalysis(getOperation());
    cir::registerCIRAliasAnalyses(aliasAnalysis);
    runAliasAnalysisOnOperation(getOperation(), aliasAnalysis);
  }
};

//===----------------------------------------------------------------------===//
// Testing ModRefResult
//===----------------------------------------------------------------------===//

struct TestCIRAliasAnalysisModRefPass
    : public test::TestAliasAnalysisModRefBase,
      PassWrapper<TestCIRAliasAnalysisModRefPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCIRAliasAnalysisModRefPass)

  StringRef getArgument() const final {
    return "test-cir-alias-analysis-modref";
  }
  StringRef getDescription() const final {
    return "Test CIR alias analysis ModRef results.";
  }
  void runOnOperation() override {
    mlir::AliasAnalysis aliasAnalysis(getOperation());
    cir::registerCIRAliasAnalyses(aliasAnalysis);
    runAliasAnalysisOnOperation(getOperation(), aliasAnalysis);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace cir {
namespace test {
void registerTestCIRAliasAnalysisPass() {
  PassRegistration<TestCIRAliasAnalysisPass>();
  PassRegistration<TestCIRAliasAnalysisModRefPass>();
}
} // namespace test
} // namespace cir

//===- TestAliasAnalysis.cpp - Test FIR lias analysis     -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/test/lib/Analysis/TestAliasAnalysis.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "flang/Optimizer/Analysis/AliasAnalysis.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Testing AliasResult
//===----------------------------------------------------------------------===//

struct TestFIRAliasAnalysisPass
    : public test::TestAliasAnalysisBase,
      PassWrapper<TestFIRAliasAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFIRAliasAnalysisPass)

  StringRef getArgument() const final { return "test-fir-alias-analysis"; }
  StringRef getDescription() const final {
    return "Test alias analysis results.";
  }
  void runOnOperation() override {
    mlir::AliasAnalysis aliasAnalysis(getOperation());
    aliasAnalysis.addAnalysisImplementation(fir::AliasAnalysis());
    runAliasAnalysisOnOperation(getOperation(), aliasAnalysis);
  }
};

//===----------------------------------------------------------------------===//
// Testing ModRefResult
//===----------------------------------------------------------------------===//

struct TestFIRAliasAnalysisModRefPass
    : public test::TestAliasAnalysisModRefBase,
      PassWrapper<TestFIRAliasAnalysisModRefPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFIRAliasAnalysisModRefPass)

  StringRef getArgument() const final {
    return "test-fir-alias-analysis-modref";
  }
  StringRef getDescription() const final {
    return "Test alias analysis ModRef results.";
  }
  void runOnOperation() override {
    mlir::AliasAnalysis aliasAnalysis(getOperation());
    aliasAnalysis.addAnalysisImplementation(fir::AliasAnalysis());
    runAliasAnalysisOnOperation(getOperation(), aliasAnalysis);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace fir {
namespace test {
void registerTestFIRAliasAnalysisPass() {
  PassRegistration<TestFIRAliasAnalysisPass>();
  PassRegistration<TestFIRAliasAnalysisModRefPass>();
}
} // namespace test
} // namespace fir
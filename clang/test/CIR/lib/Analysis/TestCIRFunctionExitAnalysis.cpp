//===- TestCIRFunctionExitAnalysis.cpp - Test CIR function exit analysis --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "clang/CIR/Dialect/Analysis/CIRFunctionExitAnalysis.h"

using namespace mlir;

namespace {

struct TestCIRFunctionExitAnalysisPass
    : public PassWrapper<TestCIRFunctionExitAnalysisPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCIRFunctionExitAnalysisPass)

  StringRef getArgument() const final {
    return "test-cir-function-exit-analysis";
  }
  StringRef getDescription() const final {
    return "Test CIR function exit analysis results.";
  }
  void runOnOperation() override {
    cir::analyzeFunctionExits(getOperation());
    markAllAnalysesPreserved();
  }
};

} // namespace

namespace cir {
namespace test {
void registerTestCIRFunctionExitAnalysisPass() {
  PassRegistration<TestCIRFunctionExitAnalysisPass>();
}
} // namespace test
} // namespace cir

//===- TestCIRPointsToAnalysis.cpp - Test CIR points-to analysis ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "clang/CIR/Dialect/Analysis/CIRPointsToAnalysis.h"

using namespace mlir;

namespace {

/// Compute the points-to analysis of the operation, with limits that can be
/// overridden. The results are checked through the remarks the analysis emits.
struct TestCIRPointsToAnalysisPass
    : PassWrapper<TestCIRPointsToAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCIRPointsToAnalysisPass)

  TestCIRPointsToAnalysisPass() = default;
  TestCIRPointsToAnalysisPass(const TestCIRPointsToAnalysisPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final { return "test-cir-points-to"; }
  StringRef getDescription() const final {
    return "Test CIR points-to analysis results.";
  }

  Option<unsigned> maxOperations{
      *this, "max-operations",
      llvm::cl::desc("Override the operation limit of the analysis"),
      llvm::cl::init(cir::PointsToLimits().maxOperations)};
  Option<unsigned> minWork{
      *this, "min-work",
      llvm::cl::desc("Override the minimum work limit of the analysis"),
      llvm::cl::init(cir::PointsToLimits().minWork)};
  Option<unsigned> workPerOperation{
      *this, "work-per-operation",
      llvm::cl::desc("Override the work limit for each operation"),
      llvm::cl::init(cir::PointsToLimits().workPerOperation)};
  Option<unsigned> maxSetSize{
      *this, "max-set-size",
      llvm::cl::desc("Override the set size limit of the analysis"),
      llvm::cl::init(cir::PointsToLimits().maxSetSize)};

  void runOnOperation() override {
    cir::PointsToLimits limits;
    limits.maxOperations = maxOperations;
    limits.minWork = minWork;
    limits.workPerOperation = workPerOperation;
    limits.maxSetSize = maxSetSize;

    cir::PointsToAnalysis pointsTo(getOperation(), limits);
    markAllAnalysesPreserved();
  }
};

} // namespace

namespace cir {
namespace test {
void registerTestCIRPointsToAnalysisPass() {
  PassRegistration<TestCIRPointsToAnalysisPass>();
}
} // namespace test
} // namespace cir

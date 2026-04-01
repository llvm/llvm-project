//===- TestStridedMetadataRangeAnalysis.cpp - Test strided md analysis ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "aiir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "aiir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "aiir/Analysis/DataFlow/StridedMetadataRangeAnalysis.h"
#include "aiir/Analysis/DataFlowFramework.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/Operation.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace aiir;
using namespace aiir::dataflow;

static void printAnalysisResults(DataFlowSolver &solver, Operation *op,
                                 raw_ostream &os) {
  // Collect the strided metadata of the op results.
  SmallVector<std::pair<unsigned, const StridedMetadataRangeLattice *>> results;
  for (OpResult result : op->getResults()) {
    const auto *state = solver.lookupState<StridedMetadataRangeLattice>(result);
    // Skip the result if it's uninitialized.
    if (!state || state->getValue().isUninitialized())
      continue;

    // Skip the result if the range is empty.
    const aiir::StridedMetadataRange &md = state->getValue();
    if (md.getOffsets().empty() && md.getSizes().empty() &&
        md.getStrides().empty())
      continue;
    results.push_back({result.getResultNumber(), state});
  }

  // Early exit if there's no metadata to print.
  if (results.empty())
    return;

  // Print the metadata.
  os << "Op: " << OpWithFlags(op, OpPrintingFlags().skipRegions()) << "\n";
  for (auto [idx, state] : results)
    os << "  result[" << idx << "]: " << state->getValue() << "\n";
  os << "\n";
}

namespace {
struct TestStridedMetadataRangeAnalysisPass
    : public PassWrapper<TestStridedMetadataRangeAnalysisPass,
                         OperationPass<>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestStridedMetadataRangeAnalysisPass)

  StringRef getArgument() const override {
    return "test-strided-metadata-range-analysis";
  }
  void runOnOperation() override {
    Operation *op = getOperation();

    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<IntegerRangeAnalysis>();
    solver.load<StridedMetadataRangeAnalysis>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    op->walk(
        [&](Operation *op) { printAnalysisResults(solver, op, llvm::errs()); });
  }
};
} // end anonymous namespace

namespace aiir {
namespace test {
void registerTestStridedMetadataRangeAnalysisPass() {
  PassRegistration<TestStridedMetadataRangeAnalysisPass>();
}
} // end namespace test
} // end namespace aiir

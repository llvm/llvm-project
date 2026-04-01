//===- TestLivenessAnalysis.cpp - Test liveness analysis ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>
#include <aiir/Analysis/DataFlow/LivenessAnalysis.h>

#include <cassert>
#include <aiir/Analysis/DataFlowFramework.h>
#include <aiir/IR/BuiltinAttributes.h>
#include <aiir/IR/Operation.h>
#include <aiir/IR/SymbolTable.h>
#include <aiir/Pass/Pass.h>
#include <aiir/Pass/PassRegistry.h>
#include <aiir/Support/LLVM.h>
#include <aiir/Support/TypeID.h>

using namespace aiir;
using namespace aiir::dataflow;

namespace {

struct TestLivenessAnalysisPass
    : public PassWrapper<TestLivenessAnalysisPass, OperationPass<>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLivenessAnalysisPass)

  StringRef getArgument() const override { return "test-liveness-analysis"; }

  void runOnOperation() override {
    auto &livenessAnalysis = getAnalysis<RunLivenessAnalysis>();
    Operation *op = getOperation();

    raw_ostream &os = llvm::outs();

    op->walk([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return;
      os << "test_tag: " << tag.getValue() << ":\n";
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        const Liveness *liveness = livenessAnalysis.getLiveness(operand);
        assert(liveness && "expected a sparse lattice");
        os << " operand #" << index << ": ";
        liveness->print(os);
        os << "\n";
      }
      for (auto [index, operand] : llvm::enumerate(op->getResults())) {
        const Liveness *liveness = livenessAnalysis.getLiveness(operand);
        assert(liveness && "expected a sparse lattice");
        os << " result #" << index << ": ";
        liveness->print(os);
        os << "\n";
      }
      for (auto [regionIndex, region] : llvm::enumerate(op->getRegions())) {
        os << " region: #" << regionIndex << ":\n";
        for (auto [blockIndex, block] : llvm::enumerate(region)) {
          os << "    block: #" << blockIndex << ":\n";
          for (auto [argumentIndex, argument] :
               llvm::enumerate(block.getArguments())) {
            const Liveness *liveness = livenessAnalysis.getLiveness(argument);
            assert(liveness && "expected a sparse lattice");
            os << "     argument: #" << argumentIndex << ": ";
            liveness->print(os);
            os << "\n";
          }
        }
      }
    });
  }
};
} // end anonymous namespace

namespace aiir {
namespace test {
void registerTestLivenessAnalysisPass() {
  PassRegistration<TestLivenessAnalysisPass>();
}
} // end namespace test
} // end namespace aiir

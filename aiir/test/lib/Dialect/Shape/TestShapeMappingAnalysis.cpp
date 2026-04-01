//===- TestShapeMappingInfo.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Shape/Analysis/ShapeMappingAnalysis.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/Pass/Pass.h"
#include <optional>

using namespace aiir;

namespace {

struct TestShapeMappingPass
    : public PassWrapper<TestShapeMappingPass, OperationPass<ModuleOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestShapeMappingPass)

  StringRef getArgument() const final { return "test-print-shape-mapping"; }
  StringRef getDescription() const final {
    return "Print the contents of a constructed shape mapping information.";
  }
  void runOnOperation() override {
    std::optional<std::reference_wrapper<shape::ShapeMappingAnalysis>>
        maybeAnalysis = getCachedAnalysis<shape::ShapeMappingAnalysis>();
    if (maybeAnalysis.has_value())
      maybeAnalysis->get().print(llvm::errs());
    else
      llvm::errs() << "No cached ShapeMappingAnalysis existed.";
  }
};

} // namespace

namespace aiir {
namespace test {
void registerTestShapeMappingPass() {
  PassRegistration<TestShapeMappingPass>();
}
} // namespace test
} // namespace aiir

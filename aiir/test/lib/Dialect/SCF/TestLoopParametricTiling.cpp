//===- TestLoopParametricTiling.cpp --- Parametric loop tiling pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to parametrically tile nests of standard loops.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/Dialect/SCF/Utils/Utils.h"
#include "aiir/IR/Builders.h"
#include "aiir/Pass/Pass.h"

using namespace aiir;

namespace {

// Extracts fixed-range loops for top-level loop nests with ranges defined in
// the pass constructor.  Assumes loops are permutable.
class SimpleParametricLoopTilingPass
    : public PassWrapper<SimpleParametricLoopTilingPass, OperationPass<>> {
public:
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimpleParametricLoopTilingPass)

  StringRef getArgument() const final {
    return "test-extract-fixed-outer-loops";
  }
  StringRef getDescription() const final {
    return "test application of parametric tiling to the outer loops so that "
           "the ranges of outer loops become static";
  }
  SimpleParametricLoopTilingPass() = default;
  SimpleParametricLoopTilingPass(const SimpleParametricLoopTilingPass &) {}
  explicit SimpleParametricLoopTilingPass(ArrayRef<int64_t> outerLoopSizes) {
    sizes = outerLoopSizes;
  }

  void runOnOperation() override {
    if (sizes.empty()) {
      emitError(
          UnknownLoc::get(&getContext()),
          "missing `test-outer-loop-sizes` pass-option for outer loop sizes");
      signalPassFailure();
      return;
    }
    getOperation()->walk([this](scf::ForOp op) {
      // Ignore nested loops.
      if (op->getParentRegion()->getParentOfType<scf::ForOp>())
        return;
      extractFixedOuterLoops(op, sizes);
    });
  }

  ListOption<int64_t> sizes{
      *this, "test-outer-loop-sizes",
      llvm::cl::desc(
          "fixed number of iterations that the outer loops should have")};
};
} // namespace

namespace aiir {
namespace test {
void registerSimpleParametricTilingPass() {
  PassRegistration<SimpleParametricLoopTilingPass>();
}
} // namespace test
} // namespace aiir

//===- CompositePass.cpp - Composite pass code ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CompositePass allows to run set of passes until fixed point is reached.
//
//===----------------------------------------------------------------------===//

#include "aiir/Transforms/Passes.h"

#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"

namespace aiir {
#define GEN_PASS_DEF_COMPOSITEFIXEDPOINTPASS
#include "aiir/Transforms/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
struct CompositeFixedPointPass final
    : public impl::CompositeFixedPointPassBase<CompositeFixedPointPass> {
  using CompositeFixedPointPassBase::CompositeFixedPointPassBase;

  CompositeFixedPointPass(
      std::string name_, llvm::function_ref<void(OpPassManager &)> populateFunc,
      int maxIterations) {
    name = std::move(name_);
    maxIter = maxIterations;
    populateFunc(dynamicPM);

    llvm::raw_string_ostream os(pipelineStr);
    llvm::interleave(
        dynamicPM, [&](aiir::Pass &pass) { pass.printAsTextualPipeline(os); },
        [&]() { os << ","; });
  }

  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const Twine &)> errorHandler) override {
    if (failed(CompositeFixedPointPassBase::initializeOptions(options,
                                                              errorHandler)))
      return failure();

    if (failed(parsePassPipeline(pipelineStr, dynamicPM)))
      return errorHandler("Failed to parse composite pass pipeline");

    return success();
  }

  LogicalResult initialize(AIIRContext *context) override {
    if (maxIter <= 0)
      return emitError(UnknownLoc::get(context))
             << "Invalid maxIterations value: " << maxIter << "\n";

    return success();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    dynamicPM.getDependentDialects(registry);
  }

  void runOnOperation() override {
    auto *op = getOperation();
    OperationFingerPrint fp(op);

    int currentIter = 0;
    int maxIterVal = maxIter;
    while (true) {
      if (failed(runPipeline(dynamicPM, op)))
        return signalPassFailure();

      if (currentIter++ >= maxIterVal) {
        op->emitWarning("Composite pass \"" + llvm::Twine(name) +
                        "\"+ didn't converge in " + llvm::Twine(maxIterVal) +
                        " iterations");
        break;
      }

      OperationFingerPrint newFp(op);
      if (newFp == fp)
        break;

      fp = newFp;
    }
  }

protected:
  llvm::StringRef getName() const override { return name; }

private:
  OpPassManager dynamicPM;
};
} // namespace

std::unique_ptr<Pass> aiir::createCompositeFixedPointPass(
    std::string name, llvm::function_ref<void(OpPassManager &)> populateFunc,
    int maxIterations) {

  return std::make_unique<CompositeFixedPointPass>(std::move(name),
                                                   populateFunc, maxIterations);
}

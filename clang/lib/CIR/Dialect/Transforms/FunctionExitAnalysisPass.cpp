//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/CIR/Dialect/Analysis/CIRFunctionExitAnalysis.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_FUNCTIONEXITANALYSIS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct FunctionExitAnalysisPass
    : public impl::FunctionExitAnalysisBase<FunctionExitAnalysisPass> {
  void runOnOperation() override {
    for (cir::FunctionExitResult result :
         cir::analyzeFunctionExits(getOperation()))
      emitRemark(result.func.getLoc())
          << "function exit: " << cir::stringifyFunctionExitKind(result.kind);
    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createFunctionExitAnalysisPass() {
  return std::make_unique<FunctionExitAnalysisPass>();
}

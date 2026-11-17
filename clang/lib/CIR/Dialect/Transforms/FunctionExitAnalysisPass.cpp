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

/// A plain description of `kind`, for the remark.
static llvm::StringRef describe(cir::FunctionExitKind kind) {
  switch (kind) {
  case cir::FunctionExitKind::Unknown:
    return "the body could not be analyzed";
  case cir::FunctionExitKind::ReturnOnly:
    return "control returns but never reaches the end of the body";
  case cir::FunctionExitKind::NoReturn:
    return "control never returns or reaches the end of the body";
  case cir::FunctionExitKind::MayReachEnd:
    return "control reaches the end of the body on some paths";
  case cir::FunctionExitKind::MustReachEnd:
    return "control leaves only by reaching the end of the body";
  }
  llvm_unreachable("unknown FunctionExitKind");
}

namespace {

struct FunctionExitAnalysisPass
    : public impl::FunctionExitAnalysisBase<FunctionExitAnalysisPass> {
  void runOnOperation() override {
    for (cir::FunctionExitResult result :
         cir::analyzeFunctionExits(getOperation()))
      emitRemark(result.func.getLoc())
          << "function exit: " << describe(result.kind) << " ["
          << cir::stringifyFunctionExitKind(result.kind) << "]";
    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createFunctionExitAnalysisPass() {
  return std::make_unique<FunctionExitAnalysisPass>();
}

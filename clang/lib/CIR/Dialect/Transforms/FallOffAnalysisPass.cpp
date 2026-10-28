//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/CIR/Dialect/Analysis/CIRFallOffAnalysis.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_FALLOFFANALYSIS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct FallOffAnalysisPass
    : public impl::FallOffAnalysisBase<FallOffAnalysisPass> {
  void runOnOperation() override {
    for (cir::FallOffResult result : cir::analyzeFallOff(getOperation()))
      emitRemark(result.func.getLoc())
          << "fall-off: " << cir::stringifyFallOffKind(result.kind);
    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createFallOffAnalysisPass() {
  return std::make_unique<FallOffAnalysisPass>();
}

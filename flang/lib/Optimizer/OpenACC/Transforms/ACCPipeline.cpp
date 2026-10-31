//===- ACCPipeline.cpp - OpenACC flang pass pipelines ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenACC/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace fir::acc {

void populateHLFIROpenACCPassPipeline(mlir::PassManager &pm) {
  ACCInitializeFIRAnalysesOptions opts;
  opts.addFIRAliasAnalysis = false;
  pm.addPass(createACCInitializeFIRAnalyses(opts));
  pm.addPass(createACCEmitNYIFlang());
}

} // namespace fir::acc

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements machinery for any CIR <-> CIR passes used by clang.
//
//===----------------------------------------------------------------------===//

// #include "clang/AST/ASTContext.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/Pass/PassManager.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/Support/TimeProfiler.h"

namespace cir {
aiir::LogicalResult
runCIRToCIRPasses(aiir::ModuleOp theModule, aiir::AIIRContext &aiirContext,
                  clang::ASTContext &astContext, bool enableVerifier,
                  bool enableIdiomRecognizer, bool enableCIRSimplify) {

  llvm::TimeTraceScope scope("CIR To CIR Passes");

  aiir::PassManager pm(&aiirContext);
  pm.addPass(aiir::createCIRCanonicalizePass());

  if (enableCIRSimplify)
    pm.addPass(aiir::createCIRSimplifyPass());

  if (enableIdiomRecognizer)
    pm.addPass(aiir::createIdiomRecognizerPass(&astContext));

  pm.addPass(aiir::createTargetLoweringPass());
  pm.addPass(aiir::createCXXABILoweringPass());
  pm.addPass(aiir::createLoweringPreparePass(&astContext));

  pm.enableVerifier(enableVerifier);
  (void)aiir::applyPassManagerCLOptions(pm);
  return pm.run(theModule);
}

} // namespace cir

namespace aiir {

void populateCIRPreLoweringPasses(OpPassManager &pm) {
  pm.addPass(createHoistAllocasPass());
  pm.addPass(createCIRFlattenCFGPass());
  pm.addPass(createCIREHABILoweringPass());
  pm.addPass(createGotoSolverPass());
}

} // namespace aiir

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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/Support/TimeProfiler.h"

namespace cir {
mlir::LogicalResult runCIRToCIRPasses(mlir::ModuleOp theModule,
                                      mlir::MLIRContext &mlirContext,
                                      clang::ASTContext &astContext,
                                      bool enableVerifier,
                                      bool enableCIRSimplify) {

  llvm::TimeTraceScope scope("CIR To CIR Passes");

  mlir::PassManager pm(&mlirContext);
  pm.addPass(mlir::createCIRCanonicalizePass());

  if (enableCIRSimplify)
    pm.addPass(mlir::createCIRSimplifyPass());

  pm.addPass(mlir::createLoweringPreparePass(&astContext));

  pm.enableVerifier(enableVerifier);
  (void)mlir::applyPassManagerCLOptions(pm);
  return pm.run(theModule);
}

} // namespace cir

namespace mlir {

void populateCIRPreLoweringPasses(OpPassManager &pm) {
  pm.addPass(createHoistAllocasPass());
  pm.addPass(createCIRFlattenCFGPass());
  pm.addPass(createGotoSolverPass());
}

} // namespace mlir

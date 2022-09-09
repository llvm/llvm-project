//====- CIRPasses.cpp - Lowering from CIR to LLVM -------------------------===//
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

#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "mlir/Dialect/CIR/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace cir {
void runCIRToCIRPasses(mlir::ModuleOp theModule, mlir::MLIRContext *mlirCtx,
                       bool enableVerifier) {
  mlir::PassManager pm(mlirCtx);
  pm.addPass(mlir::createMergeCleanupsPass());
  pm.enableVerifier(enableVerifier);

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    llvm::report_fatal_error(
        "CIR codegen: MLIR pass manager fails when running CIR passes!");
}
} // namespace cir
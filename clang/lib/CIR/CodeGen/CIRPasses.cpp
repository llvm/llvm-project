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

#include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

namespace cir {
mlir::LogicalResult runCIRToCIRPasses(mlir::ModuleOp theModule,
                                      mlir::MLIRContext *mlirCtx,
                                      clang::ASTContext &astCtx,
                                      bool enableVerifier, bool enableLifetime,
                                      llvm::StringRef lifetimeOpts,
                                      llvm::StringRef idiomRecognizerOpts,
                                      bool &passOptParsingFailure) {
  mlir::PassManager pm(mlirCtx);
  passOptParsingFailure = false;

  pm.addPass(mlir::createMergeCleanupsPass());

  // TODO(CIR): Make this actually propagate errors correctly. This is stubbed
  // in to get rebases going.
  auto errorHandler = [](const llvm::Twine &) -> mlir::LogicalResult {
    return mlir::LogicalResult::failure();
  };

  if (enableLifetime) {
    auto lifetimePass = mlir::createLifetimeCheckPass(&astCtx);
    if (lifetimePass->initializeOptions(lifetimeOpts, errorHandler).failed()) {
      passOptParsingFailure = true;
      return mlir::failure();
    }
    pm.addPass(std::move(lifetimePass));
  }

  auto idiomPass = mlir::createIdiomRecognizerPass(&astCtx);
  if (idiomPass->initializeOptions(idiomRecognizerOpts, errorHandler)
          .failed()) {
    passOptParsingFailure = true;
    return mlir::failure();
  }
  pm.addPass(std::move(idiomPass));

  pm.addPass(mlir::createLoweringPreparePass(&astCtx));

  // FIXME: once CIRCodenAction fixes emission other than CIR we
  // need to run this right before dialect emission.
  pm.addPass(mlir::createDropASTPass());
  pm.enableVerifier(enableVerifier);
  (void)mlir::applyPassManagerCLOptions(pm);
  return pm.run(theModule);
}
} // namespace cir

//===- Passes.h - CIR pass entry points -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_PASSES_H
#define CLANG_CIR_DIALECT_PASSES_H

#include "mlir/Pass/Pass.h"

namespace clang {
class ASTContext;
}
namespace mlir {

std::unique_ptr<Pass> createCIRCanonicalizePass();
std::unique_ptr<Pass> createCIRFlattenCFGPass();
std::unique_ptr<Pass> createCIRSimplifyPass();
std::unique_ptr<Pass> createHoistAllocasPass();
std::unique_ptr<Pass> createLoweringPreparePass();
std::unique_ptr<Pass> createLoweringPreparePass(clang::ASTContext *astCtx);
std::unique_ptr<Pass> createGotoSolverPass();

void populateCIRPreLoweringPasses(mlir::OpPassManager &pm);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerCIRDialectTranslation(mlir::MLIRContext &context);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "clang/CIR/Dialect/Passes.h.inc"

} // namespace mlir

#endif // CLANG_CIR_DIALECT_PASSES_H

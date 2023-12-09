//====- CIRToCIRPasses.h- Lowering from CIR to LLVM -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an interface for converting CIR modules to LLVM IR.
//
//===----------------------------------------------------------------------===//
#ifndef CLANG_CIR_CIRTOCIRPASSES_H
#define CLANG_CIR_CIRTOCIRPASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace clang {
class ASTContext;
}

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace cir {

// Run set of cleanup/prepare/etc passes CIR <-> CIR.
mlir::LogicalResult runCIRToCIRPasses(mlir::ModuleOp theModule,
                                      mlir::MLIRContext *mlirCtx,
                                      clang::ASTContext &astCtx,
                                      bool enableVerifier, bool enableLifetime,
                                      llvm::StringRef lifetimeOpts,
                                      llvm::StringRef idiomRecognizerOpts,
                                      bool &passOptParsingFailure);
} // namespace cir

#endif // CLANG_CIR_CIRTOCIRPASSES_H_

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an interface for running CIR-to-CIR passes.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_CIRTOCIRPASSES_H
#define CLANG_CIR_CIRTOCIRPASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

#include <memory>

namespace clang {
class ASTContext;
}

namespace llvm::vfs {
class FileSystem;
} // namespace llvm::vfs

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace cir {

class LowerModule;

// Run set of cleanup/prepare/etc passes CIR <-> CIR. The caller owns
// `lowerModule`, which provides the target/LangOpts state for AST-free
// passes (LoweringPrepare and any future helpers). `astCtx` is still
// required by IdiomRecognizer and the AST-fact materialization pass.
mlir::LogicalResult
runCIRToCIRPasses(mlir::ModuleOp theModule, mlir::MLIRContext &mlirCtx,
                  clang::ASTContext &astCtx, cir::LowerModule &lowerModule,
                  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs,
                  bool enableVerifier, bool enableIdiomRecognizer,
                  bool enableCIRSimplify, bool enableLibOpt,
                  llvm::StringRef libOptOptions, bool enableCallConvLowering);

} // namespace cir

#endif // CLANG_CIR_CIRTOCIRPASSES_H_

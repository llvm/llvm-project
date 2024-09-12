//===- Passes.h - CIR Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for ClangIR.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_PASSES_H
#define CLANG_CIR_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace cir {
/// Create a pass for lowering from MLIR builtin dialects such as `Affine` and
/// `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createConvertMLIRToLLVMPass();

/// Create a pass that fully lowers CIR to the MLIR in-tree dialects.
std::unique_ptr<mlir::Pass> createConvertCIRToMLIRPass();

namespace direct {
/// Create a pass that fully lowers CIR to the LLVMIR dialect.
std::unique_ptr<mlir::Pass> createConvertCIRToLLVMPass();

/// Adds passes that fully lower CIR to the LLVMIR dialect.
void populateCIRToLLVMPasses(mlir::OpPassManager &pm);

} // namespace direct
} // end namespace cir

#endif // CLANG_CIR_PASSES_H

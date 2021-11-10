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

/// Create a pass for lowering from `CIR` operations well as `Affine` and `Std`,
/// to the LLVM dialect for codegen. We'll want to separate this eventually into
/// different phases instead of doing it all at once.
std::unique_ptr<mlir::Pass> createConvertCIRToLLVMPass();
} // end namespace cir

#endif // CLANG_CIR_PASSES_H

//===----------------------------------------------------------------------===//
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
namespace direct {
/// Create a pass that fully lowers CIR to the LLVMIR dialect.
std::unique_ptr<mlir::Pass> createConvertCIRToLLVMPass();

/// Adds passes that fully lower CIR to the LLVMIR dialect. When enableOpenMP
/// is set (-fopenmp), the OpenMP lowering passes are also added.
void populateCIRToLLVMPasses(mlir::OpPassManager &pm, bool enableOpenMP);

} // namespace direct
} // end namespace cir

#endif // CLANG_CIR_PASSES_H

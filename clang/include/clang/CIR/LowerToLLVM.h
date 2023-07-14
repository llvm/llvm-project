//====- LowerToLLVM.h- Lowering from CIR to LLVM --------------------------===//
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
#ifndef CLANG_CIR_LOWERTOLLVM_H
#define CLANG_CIR_LOWERTOLLVM_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace cir {

namespace direct {
std::unique_ptr<llvm::Module>
lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp theModule,
                             llvm::LLVMContext &llvmCtx);
}

// Lower directly from pristine CIR to LLVMIR.
std::unique_ptr<llvm::Module>
lowerFromCIRToMLIRToLLVMIR(mlir::ModuleOp theModule,
                           std::unique_ptr<mlir::MLIRContext> mlirCtx,
                           llvm::LLVMContext &llvmCtx);

mlir::ModuleOp lowerFromCIRToMLIR(mlir::ModuleOp theModule,
                                  mlir::MLIRContext *mlirCtx);
} // namespace cir

#endif // CLANG_CIR_LOWERTOLLVM_H_

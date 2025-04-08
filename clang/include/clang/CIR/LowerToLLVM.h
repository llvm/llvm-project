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

#include <memory>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace cir {

namespace direct {
std::unique_ptr<llvm::Module>
lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp mlirModule,
                             llvm::LLVMContext &llvmCtx);
} // namespace direct
} // namespace cir

#endif // CLANG_CIR_LOWERTOLLVM_H

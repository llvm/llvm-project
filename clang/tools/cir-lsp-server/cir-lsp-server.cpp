//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A language server for ClangIR
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "clang/CIR/InitAllDialects.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  cir::registerAllDialects(registry);
  registry.insert<mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect,
                  mlir::func::FuncDialect>();
  return failed(mlir::MlirLspServerMain(argc, argv, registry));
}

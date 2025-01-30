//====- LowerToLLVM.cpp - Lowering from CIR to LLVMIR ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR operations to LLVMIR.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/LowerToLLVM.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TimeProfiler.h"

using namespace cir;
using namespace llvm;

namespace cir {
namespace direct {

std::unique_ptr<llvm::Module>
lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp MOp, LLVMContext &LLVMCtx) {
  llvm::TimeTraceScope scope("lower from CIR to LLVM directly");

  std::optional<StringRef> ModuleName = MOp.getName();
  auto M = std::make_unique<llvm::Module>(
      ModuleName ? *ModuleName : "CIRToLLVMModule", LLVMCtx);

  if (!M)
    report_fatal_error("Lowering from LLVMIR dialect to llvm IR failed!");

  return M;
}
} // namespace direct
} // namespace cir

//===-- Optimizer/CodeGen/CodeGen.h -- code generation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_CODEGEN_CODEGEN_H
#define OPTIMIZER_CODEGEN_CODEGEN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace fir {

struct NameUniquer;

/// Prerequiste pass for code gen. Perform intermediate rewrites to perform
/// the code gen (to LLVM-IR dialect) conversion.
std::unique_ptr<mlir::Pass> createFirCodeGenRewritePass();

/// Convert FIR to the LLVM IR dialect
std::unique_ptr<mlir::Pass> createFIRToLLVMPass();

/// Convert the LLVM IR dialect to LLVM-IR proper
std::unique_ptr<mlir::Pass>
createLLVMDialectToLLVMPass(llvm::raw_ostream &output);

// declarative passes
#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"

} // namespace fir

#endif // OPTIMIZER_CODEGEN_CODEGEN_H

//===-- Optimizer/CodeGen/CodeGen.h -- code generation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_CODEGEN_CODEGEN_H
#define FORTRAN_OPTIMIZER_CODEGEN_CODEGEN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace fir {

struct NameUniquer;

/// Prerequiste pass for code gen. Perform intermediate rewrites to perform
/// the code gen (to LLVM-IR dialect) conversion.
std::unique_ptr<mlir::Pass> createFirCodeGenRewritePass();

/// FirTargetRewritePass options.
struct TargetRewriteOptions {
  bool noCharacterConversion{};
  bool noComplexConversion{};
};

/// Prerequiste pass for code gen. Perform intermediate rewrites to tailor the
/// FIR for the chosen target.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createFirTargetRewritePass(
    const TargetRewriteOptions &options = TargetRewriteOptions());

/// Convert FIR to the LLVM IR dialect
std::unique_ptr<mlir::Pass> createFIRToLLVMPass();

using LLVMIRLoweringPrinter =
    std::function<void(llvm::Module &, llvm::raw_ostream &)>;
/// Convert the LLVM IR dialect to LLVM-IR proper
std::unique_ptr<mlir::Pass> createLLVMDialectToLLVMPass(
    llvm::raw_ostream &output,
    LLVMIRLoweringPrinter printer =
        [](llvm::Module &m, llvm::raw_ostream &out) { m.print(out, nullptr); });

// declarative passes
#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"

} // namespace fir

#endif // FORTRAN_OPTIMIZER_CODEGEN_CODEGEN_H

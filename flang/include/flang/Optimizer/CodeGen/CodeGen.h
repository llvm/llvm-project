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

#define GEN_PASS_DECL_FIRTOLLVMLOWERING
#define GEN_PASS_DECL_CODEGENREWRITE
#define GEN_PASS_DECL_TARGETREWRITEPASS
#define GEN_PASS_DECL_BOXEDPROCEDUREPASS
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"

/// Prerequiste pass for code gen. Perform intermediate rewrites to perform
/// the code gen (to LLVM-IR dialect) conversion.
std::unique_ptr<mlir::Pass> createFirCodeGenRewritePass();

/// FirTargetRewritePass options.
struct TargetRewriteOptions {
  bool noCharacterConversion{};
  bool noComplexConversion{};
  bool noStructConversion{};
};

/// Prerequiste pass for code gen. Perform intermediate rewrites to tailor the
/// FIR for the chosen target.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createFirTargetRewritePass(
    const TargetRewriteOptions &options = TargetRewriteOptions());

/// FIR to LLVM translation pass options.
struct FIRToLLVMPassOptions {
  // Do not fail when type descriptors are not found when translating
  // operations that use them at the LLVM level like fir.embox. Instead,
  // just use a null pointer.
  // This is useful to test translating programs manually written where a
  // frontend did not generate type descriptor data structures. However, note
  // that such programs would crash at runtime if the derived type descriptors
  // are required by the runtime, so this is only an option to help debugging.
  bool ignoreMissingTypeDescriptors = false;

  // Generate TBAA information for FIR types and memory accessing operations.
  bool applyTBAA = false;

  // Force the usage of a unified tbaa tree in TBAABuilder.
  bool forceUnifiedTBAATree = false;
};

/// Convert FIR to the LLVM IR dialect with default options.
std::unique_ptr<mlir::Pass> createFIRToLLVMPass();

/// Convert FIR to the LLVM IR dialect
std::unique_ptr<mlir::Pass> createFIRToLLVMPass(FIRToLLVMPassOptions options);

using LLVMIRLoweringPrinter =
    std::function<void(llvm::Module &, llvm::raw_ostream &)>;

/// Convert the LLVM IR dialect to LLVM-IR proper
std::unique_ptr<mlir::Pass> createLLVMDialectToLLVMPass(
    llvm::raw_ostream &output,
    LLVMIRLoweringPrinter printer =
        [](llvm::Module &m, llvm::raw_ostream &out) { m.print(out, nullptr); });

/// Convert boxproc values to a lower level representation. The default is to
/// use function pointers and thunks.
std::unique_ptr<mlir::Pass> createBoxedProcedurePass();
std::unique_ptr<mlir::Pass> createBoxedProcedurePass(bool useThunks);

// declarative passes
#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"

} // namespace fir

#endif // FORTRAN_OPTIMIZER_CODEGEN_CODEGEN_H

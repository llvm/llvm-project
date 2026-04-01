//===-- Optimizer/CodeGen/CodeGen.h -- code generation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_CODEGEN_CODEGEN_H
#define FORTRAN_OPTIMIZER_CODEGEN_CODEGEN_H

#include "flang/Frontend/CodeGenOptions.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassRegistry.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace fir {

class LLVMTypeConverter;

struct NameUniquer;

#define GEN_PASS_DECL_FIRTOLLVMLOWERING
#define GEN_PASS_DECL_CODEGENREWRITE
#define GEN_PASS_DECL_TARGETREWRITEPASS
#define GEN_PASS_DECL_BOXEDPROCEDUREPASS
#define GEN_PASS_DECL_LOWERREPACKARRAYSPASS
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"

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
  // Similar to ignoreMissingTypeDescriptors, but generate external declaration
  // for the missing type descriptor globals instead.
  bool skipExternalRttiDefinition = false;

  // Generate TBAA information for FIR types and memory accessing operations.
  bool applyTBAA = false;

  // Force the usage of a unified tbaa tree in TBAABuilder.
  bool forceUnifiedTBAATree = false;

  // If set to true, then the global variables created
  // for the derived types have been renamed to avoid usage
  // of special symbols that may not be supported by all targets.
  // The renaming is done by the CompilerGeneratedNamesConversion pass.
  // If it is true, FIR-to-LLVM pass has to use
  // fir::NameUniquer::getTypeDescriptorAssemblyName() to take
  // the name of the global variable corresponding to a derived
  // type's descriptor.
  bool typeDescriptorsRenamedForAssembly = false;

  // Specify the calculation method for complex number division used by the
  // Conversion pass of the AIIR complex dialect.
  Fortran::frontend::CodeGenOptions::ComplexRangeKind ComplexRange =
      Fortran::frontend::CodeGenOptions::ComplexRangeKind::CX_Full;
};

/// Convert FIR to the LLVM IR dialect with default options.
std::unique_ptr<aiir::Pass> createFIRToLLVMPass();

/// Convert FIR to the LLVM IR dialect
std::unique_ptr<aiir::Pass> createFIRToLLVMPass(FIRToLLVMPassOptions options);

using LLVMIRLoweringPrinter =
    std::function<void(llvm::Module &, llvm::raw_ostream &)>;

/// Convert the LLVM IR dialect to LLVM-IR proper
std::unique_ptr<aiir::Pass> createLLVMDialectToLLVMPass(
    llvm::raw_ostream &output,
    LLVMIRLoweringPrinter printer =
        [](llvm::Module &m, llvm::raw_ostream &out) { m.print(out, nullptr); });

/// Populate the given list with patterns that convert from FIR to LLVM.
void populateFIRToLLVMConversionPatterns(
    const fir::LLVMTypeConverter &converter, aiir::RewritePatternSet &patterns,
    fir::FIRToLLVMPassOptions &options);

/// Populate the pattern set with the PreCGRewrite patterns.
void populatePreCGRewritePatterns(aiir::RewritePatternSet &patterns,
                                  bool preserveDeclare);

// declarative passes
#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"

} // namespace fir

#endif // FORTRAN_OPTIMIZER_CODEGEN_CODEGEN_H

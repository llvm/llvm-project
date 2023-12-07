//===-- CGPasses.td - code gen pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Optimizer/CodeGen/
//  directory.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_CODEGEN_FIR_PASSES
#define FORTRAN_OPTIMIZER_CODEGEN_FIR_PASSES

include "mlir/Pass/PassBase.td"

def FIRToLLVMLowering : Pass<"fir-to-llvm-ir", "mlir::ModuleOp"> {
  let summary = "Convert FIR dialect to LLVM-IR dialect";
  let description = [{
    Convert the FIR dialect to the LLVM-IR dialect of MLIR. This conversion
    will also convert ops in the standard and FIRCG dialects.
  }];
  let constructor = "::fir::createFIRToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
  let options = [
    Option<"forcedTargetTriple", "target", "std::string", /*default=*/"",
           "Override module's target triple.">,
    Option<"applyTBAA", "apply-tbaa", "bool", /*default=*/"false",
           "Attach TBAA tags to memory accessing operations.">
  ];
}

def CodeGenRewrite : Pass<"cg-rewrite", "mlir::ModuleOp"> {
  let summary = "Rewrite some FIR ops into their code-gen forms.";
  let description = [{
    Fuse specific subgraphs into single Ops for code generation.
  }];
  let constructor = "::fir::createFirCodeGenRewritePass()";
  let dependentDialects = [
    "fir::FIROpsDialect", "fir::FIRCodeGenDialect"
  ];
  let statistics = [
    Statistic<"numDCE", "num-dce'd", "Number of operations eliminated">
  ];
}

def TargetRewritePass : Pass<"target-rewrite", "mlir::ModuleOp"> {
  let summary = "Rewrite some FIR dialect into target specific forms.";
  let description = [{
      Certain abstractions in the FIR dialect need to be rewritten to reflect
      representations that may differ based on the target machine.
  }];
  let constructor = "::fir::createFirTargetRewritePass()";
  let dependentDialects = [ "fir::FIROpsDialect", "mlir::func::FuncDialect" ];
  let options = [
    Option<"forcedTargetTriple", "target", "std::string", /*default=*/"",
           "Override module's target triple.">,
    Option<"noCharacterConversion", "no-character-conversion",
           "bool", /*default=*/"false",
           "Disable target-specific conversion of CHARACTER.">,
    Option<"noComplexConversion", "no-complex-conversion",
           "bool", /*default=*/"false",
           "Disable target-specific conversion of COMPLEX.">
  ];
}

def BoxedProcedurePass : Pass<"boxed-procedure", "mlir::ModuleOp"> {
  let constructor = "::fir::createBoxedProcedurePass()";
  let options = [
    Option<"useThunks", "use-thunks",
           "bool", /*default=*/"true",
           "Convert procedure pointer abstractions to a single code pointer, "
           "deploying thunks wherever required.">
  ];
}

#endif // FORTRAN_OPTIMIZER_CODEGEN_FIR_PASSES

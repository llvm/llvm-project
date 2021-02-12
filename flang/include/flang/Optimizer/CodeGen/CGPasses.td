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

def CodeGenRewrite : Pass<"cg-rewrite", "mlir::ModuleOp"> {
  let summary = "Rewrite some FIR ops into their code-gen forms.";
  let description = [{
    Fuse specific subgraphs into single Ops for code generation.
  }];
  let constructor = "fir::createFirCodeGenRewritePass()";
  let dependentDialects = [
    "fir::FIROpsDialect", "fir::FIRCodeGenDialect"
  ];
}

def TargetRewrite : Pass<"target-rewrite", "mlir::ModuleOp"> {
  let summary = "Rewrite some FIR dialect into target specific forms. "
                "Certain abstractions in the FIR dialect need to be rewritten "
                "to reflect representations that may differ based on the "
                "target machine.";
  let constructor = "fir::createFirTargetRewritePass()";
  let dependentDialects = [
    "fir::FIROpsDialect", "fir::FIRCodeGenDialect"
  ];
  let options = [
    Option<"noCharacterConversion", "no-character-conversion",
           "bool", /*default=*/"false",
           "Disable target-specific conversion of CHARACTER.">,
    Option<"noComplexConversion", "no-complex-conversion",
           "bool", /*default=*/"false",
           "Disable target-specific conversion of COMPLEX.">
  ];
}

#endif // FORTRAN_OPTIMIZER_CODEGEN_FIR_PASSES

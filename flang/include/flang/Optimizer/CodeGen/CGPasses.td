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

#ifndef FLANG_OPTIMIZER_CODEGEN_PASSES
#define FLANG_OPTIMIZER_CODEGEN_PASSES

include "mlir/Pass/PassBase.td"

def CodeGenRewrite : FunctionPass<"cg-rewrite"> {
  let summary = "Rewrite some FIR ops into their code-gen forms.";
  let constructor = "fir::createFirCodeGenRewritePass()";
}

#endif // FLANG_OPTIMIZER_CODEGEN_PASSES

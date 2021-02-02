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

def CodeGenRewrite : Pass<"cg-rewrite", "mlir::ModuleOp"> {
  let summary = "Rewrite some FIR ops into their code-gen forms.";
  let description = [{
    Fuse specific subgraphs into single Ops for code generation.
  }];
  let constructor = "fir::createFirCodeGenRewritePass()";
  let dependentDialects = ["fir::FIROpsDialect"];
}

#endif // FLANG_OPTIMIZER_CODEGEN_PASSES

//===-- Passes.td - CIR pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CIR_PASSES
#define MLIR_DIALECT_CIR_PASSES

include "mlir/Pass/PassBase.td"

def MergeCleanups : Pass<"cir-merge-cleanups"> {
  let summary = "Remove unnecessary branches to cleanup blocks";
  let description = [{
    Canonicalize pass is too aggressive for CIR when the pipeline is
    used for C/C++ analysis. This pass runs some rewrites for scopes,
    merging some blocks and eliminating unnecessary control-flow.
  }];
  let constructor = "mlir::createMergeCleanupsPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def LifetimeCheck : Pass<"cir-lifetime-check"> {
  let summary = "Check lifetime safety and generate diagnostics";
  let description = [{
    This pass relies on a lifetime analysis pass and uses the diagnostics
    mechanism to report to the user. It does not change any code.

    A default ctor is specified but is solely in order to make
    tablegen happy, since this pass requires the presence of an ASTContext,
    one can set that up using `mlir::createLifetimeCheckPass(clang::ASTContext &)`
    instead.
  }];
  let constructor = "mlir::createLifetimeCheckPass()";
  let dependentDialects = ["cir::CIRDialect"];

  let options = [
    ListOption<"historyList", "history", "std::string",
               "List of history styles to emit as part of diagnostics."
               " Supported styles: {all|null|invalid}", "llvm::cl::ZeroOrMore">,
    ListOption<"remarksList", "remarks", "std::string",
               "List of remark styles to enable as part of diagnostics."
               " Supported styles: {all|pset}", "llvm::cl::ZeroOrMore">,
    Option<"historyLimit", "history_limit", "unsigned", /*default=*/"1",
           "Max amount of diagnostics to emit on pointer history">
  ];
}

def DropAST : Pass<"cir-drop-ast"> {
  let summary = "Remove clang AST nodes attached to CIR operations";
  let description = [{
    Some CIR operations have references back to Clang AST, this is
    necessary to perform lots of useful checks without having to
    duplicate all rich AST information in CIR. As we move down in the
    pipeline (e.g. generating LLVM or other MLIR dialects), the need
    for such nodes diminish and AST information can be dropped.

    Right now this is enabled by default in Clang prior to dialect
    codegen from CIR, but not before lifetime check, where AST is
    required to be present.
  }];
  let constructor = "mlir::createDropASTPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def LoweringPrepare : Pass<"cir-lowering-prepare"> {
  let summary = "Preparation work before lowering to LLVM dialect";
  let description = [{
    This pass does preparation work for LLVM lowering. For example, it may
    expand the global variable initialziation in a more ABI-friendly form.
  }];
  let constructor = "mlir::createLoweringPreparePass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def IdiomRecognizer : Pass<"cir-idiom-recognizer"> {
  let summary = "Raise calls to C/C++ libraries to CIR operations";
  let description = [{
    This pass recognize idiomatic C++ usage and incorporate C++ standard
    containers, library functions calls, and types into CIR operation,
    attributes and types.

    Detections done by this pass can be inspected by users by using
    remarks. Currently supported are `all` and `found-calls`.
  }];
  let constructor = "mlir::createIdiomRecognizerPass()";
  let dependentDialects = ["cir::CIRDialect"];

  let options = [
    ListOption<"remarksList", "remarks", "std::string",
               "Diagnostic remarks to enable"
               " Supported styles: {all|found-calls}", "llvm::cl::ZeroOrMore">,
  ];
}

#endif // MLIR_DIALECT_CIR_PASSES

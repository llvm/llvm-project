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

def CIRSimplify : Pass<"cir-simplify"> {
  let summary = "Performs CIR simplification";
  let description = [{
    The pass rewrites CIR and removes some redundant operations.

    For example, due to canonicalize pass is too aggressive for CIR when
    the pipeline is used for C/C++ analysis, this pass runs some rewrites
    for scopes, merging some blocks and eliminating unnecessary control-flow.

    Also, the pass removes redundant and/or unneccessary cast and unary not
    operation e.g.
    ```mlir
      %1 = cir.cast(bool_to_int, %0 : !cir.bool), !s32i
      %2 = cir.cast(int_to_bool, %1 : !s32i), !cir.bool
    ```

  }];
  let constructor = "mlir::createCIRSimplifyPass()";
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

def SCFPrepare : Pass<"cir-mlir-scf-prepare"> {
  let summary = "Preparation work before lowering to SCF dialect";
  let description = [{
    This pass does preparation work for SCF lowering. For example, it may
    hoist the loop invariant or canonicalize the loop comparison. Currently,
    the pass only be enabled for through MLIR pipeline.
  }];
  let constructor = "mlir::createSCFPreparePass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def FlattenCFG : Pass<"cir-flatten-cfg"> {
  let summary = "Produces flatten cfg";
  let description = [{ 
    This pass transforms CIR and inline all the nested regions. Thus,
    the next post condtions are met after the pass applied:
    - there is not any nested region in a function body
    - all the blocks in a function belong to the parent region
    In other words, this pass removes such CIR operations like IfOp, LoopOp,
    ScopeOp and etc. and produces a flat CIR.
  }];
  let constructor = "mlir::createFlattenCFGPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def GotoSolver : Pass<"cir-goto-solver"> {
  let summary = "Replaces goto operatations with branches";
  let description = [{
    This pass transforms CIR and replaces goto-s with branch
    operations to the proper blocks.
  }];
  let constructor = "mlir::createGotoSolverPass()";
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

def LibOpt : Pass<"cir-lib-opt"> {
  let summary = "Optimize C/C++ library calls";
  let description = [{
    By using higher level information from `cir-idiom-recognize`, this pass
    apply transformations to CIR based on specific C/C++ library semantics.

    Transformations done by this pass can be inspected by users by using
    remarks. Currently supported are `all` and `transforms`.
  }];
  let constructor = "mlir::createLibOptPass()";
  let dependentDialects = ["cir::CIRDialect"];

  let options = [
    ListOption<"remarksList", "remarks", "std::string",
               "Diagnostic remarks to enable"
               " Supported styles: {all|transforms}", "llvm::cl::ZeroOrMore">,
  ];
}

def CallConvLowering : Pass<"cir-call-conv-lowering"> {
  let summary = "Handle calling conventions for CIR functions";
  let description = [{
    This pass lowers CIR function definitions and calls according to the
    calling conventions for the target architecture. This pass is necessary
    to properly lower CIR functions to LLVM IR.
  }];
  let constructor = "mlir::createCallConvLoweringPass()";
  let dependentDialects = ["mlir::cir::CIRDialect"];
}

#endif // MLIR_DIALECT_CIR_PASSES

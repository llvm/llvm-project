//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_PASSES_TD
#define CLANG_CIR_DIALECT_PASSES_TD

include "mlir/Pass/PassBase.td"

def CIRCanonicalize : Pass<"cir-canonicalize"> {
  let summary = "Performs CIR canonicalization";
  let description = [{
    Perform canonicalizations on CIR and removes some redundant operations.

    This pass performs basic cleanup and canonicalization transformations that
    are not intended to affect CIR-to-source fidelity and high-level code
    analysis passes. Example transformations performed in this pass include
    empty scope cleanup, trivial `try` cleanup, redundant branch cleanup, etc.
    Those more "heavyweight" transformations and those transformations that
    could significantly affect CIR-to-source fidelity are performed in the
    `cir-simplify` pass.
  }];

  let constructor = "mlir::createCIRCanonicalizePass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def CIRSimplify : Pass<"cir-simplify"> {
  let summary = "Performs CIR simplification and code optimization";
  let description = [{
    The pass performs semantics-preserving code simplifications and optimizations
    on CIR while maintaining strict program correctness.

    Unlike the `cir-canonicalize` pass, these transformations may reduce the IR's
    structural similarity to the original source code as a trade-off for improved
    code quality. This can affect debugging fidelity by altering intermediate
    representations of folded expressions, hoisted operations, and other
    optimized constructs.

    Example transformations include ternary expression folding and code hoisting
    while preserving program semantics.
  }];
  let constructor = "mlir::createCIRSimplifyPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def HoistAllocas : Pass<"cir-hoist-allocas"> {
  let summary = "Hoist allocas to the entry of the function";
  let description = [{
    This pass hoist all non-dynamic allocas to the entry of the function.
    This is helpful for later code generation.
  }];
  let constructor = "mlir::createHoistAllocasPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def CIRFlattenCFG : Pass<"cir-flatten-cfg"> {
  let summary = "Produces flatten CFG";
  let description = [{
    This pass transforms CIR by inlining all the nested regions. Thus,
    the following conditions are true after the pass applied:
    - there are no nested regions in any function body
    - all the blocks in a function belong to the parent region
    In other words, this pass removes such CIR operations like IfOp, LoopOp,
    ScopeOp and etc. and produces a flat CIR.
  }];
  let constructor = "mlir::createCIRFlattenCFGPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def GotoSolver : Pass<"cir-goto-solver"> {
  let summary = "Replaces goto operations with branches";
  let description = [{
    This pass transforms CIR and replaces goto-s with branch
    operations to the proper blocks.
  }];
  let constructor = "mlir::createGotoSolverPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def CXXABILowering : Pass<"cir-cxxabi-lowering", "mlir::ModuleOp"> {
  let summary = "Lower CIR according to C++ ABI requirements";
  let description = [{
    This pass lowers CIR operations and types that represent high-level C/C++
    constructs to a more fundamental form according to the target ABI
    requirements.

    See the description of the `TargetLowering` pass for the difference between
    this pass and the `TargetLowering` pass.
  }];
  let constructor = "mlir::createCXXABILoweringPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def TargetLowering : Pass<"cir-target-lowering", "mlir::ModuleOp"> {
  let summary = "Lower CIR to a target-specific form";
  let description = [{
    This pass lowers CIR operations from a target-agnostic form to a
    target-specific form without considering ABI requirements.

    CIR has three passes in its lowering pipeline that transform input CIR
    according to target-specific requirements, scheduled in the pipeline by the
    following order:

      1. The `TargetLowering` pass.
      2. The `CXXABILowering` pass.
      3. The `CallConvLowering` pass (not implemented yet).

    The `TargetLowering` pass acts more like a legalization pass. It ensures
    every operation in CIR conforms to the target's constraints. An example
    would be the handling of synchronization scopes of atomic operations. The
    x86 family of targets only support a system-wide synchronization scope, thus
    any atomic operations with a different synchronization scope would be
    transformed to use the system-wide scope in this pass.

    The `CXXABILowering` pass and the (not yet implemented) `CallConvLowering`
    pass transform the CIR according to the target's ABI requirements. The
    former handles all ABI-related lowering except for calling convention
    handling, which is handled specifically in the latter. Example
    transformations that the `CXXABILowering` pass could make include:

      - Replace C/C++ types that have an ABI-defined layout with more
        fundamental types corresponding to the ABI requirements. For example,
        the layout of the pointer-to-data-member type in C++ is ABI-defined,
        thus the `CXXABILowering` pass would replace all occurrences of this
        type according to the ABI requirements. With a typical target ABI, it
        will be replaced by the `ptrdiff_t` type since this is the type most
        ABIs use for the layout of the pointer-to-data-member type.
      - Replace CIR operations that have ABI-dependent implementations with more
        fundamental operations. For example, the `dynamic_cast` operator in C++
        is implemented in an ABI-dependent way, typically by calling into a
        library function provided by the implementation. The `CXXABILowering`
        pass would thus replace all `cir.dyn_cast` operations with corresponding
        library function calls.

    The `CallConvLowering` pass is dedicated to handle calling conventions. It
    rewrites function signatures according to calling convention requirements,
    and updates function body and call sites accordingly. For example, when
    passing a large struct by value in C/C++, most ABIs require passing a
    pointer to a copy of the struct instead. Thus, the `CallConvLowering` pass
    would rewrite function signatures that take a value of a large struct type
    to take a pointer to the struct type instead, and update the function body
    and the call sites accordingly.
  }];
  let constructor = "mlir::createTargetLoweringPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def LoweringPrepare : Pass<"cir-lowering-prepare"> {
  let summary = "Lower to more fine-grained CIR operations before lowering to "
    "other dialects";
  let description = [{
    This pass does preparation work for lowering to other dialects. For example,
    it may expand the global variable initialziation in a more ABI-friendly form.
  }];
  let constructor = "mlir::createLoweringPreparePass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def LibOpt : Pass<"cir-lib-opt"> {
  let summary = "Optimize C/C++ library calls";
  let description = [{
    This pass applies transformations on C/C++ standard library idioms,
    such as library function calls and structs raised to CIR operations
     using the `cir-idiom-recognize` pass.
  }];

  let constructor = "mlir::createLibOptPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

#endif // CLANG_CIR_DIALECT_PASSES_TD

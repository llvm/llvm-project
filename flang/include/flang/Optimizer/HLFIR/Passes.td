//===-- Passes.td - HLFIR pass definition file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_HLFIR_PASSES
#define FORTRAN_DIALECT_HLFIR_PASSES

include "mlir/Pass/PassBase.td"
def ConvertHLFIRtoFIR : Pass<"convert-hlfir-to-fir", "::mlir::ModuleOp"> {
  let summary = "Lower High-Level FIR to FIR";
  let dependentDialects = [
    "mlir::func::FuncDialect",
  ];
}

def BufferizeHLFIR : Pass<"bufferize-hlfir", "::mlir::ModuleOp"> {
  let summary = "Convert HLFIR operations operating on hlfir.expr into operations on memory";
  let options = [Option<"optimizeEmptyElementals", "opt-empty-elementals",
                        "bool", /*default=*/"false",
                        "When converting hlfir.elemental into a loop nest, "
                        "check if the resulting expression is an empty array, "
                        "and make sure none of the loops is executed.">];
}

def OptimizedBufferization : Pass<"opt-bufferization"> {
  let summary = "Special cases for hlfir.expr bufferization where we can avoid a temporary which would be created by the generic bufferization pass";
}

def LowerHLFIRIntrinsics : Pass<"lower-hlfir-intrinsics", "::mlir::ModuleOp"> {
  let summary = "Lower HLFIR transformational intrinsic operations";
}

def LowerHLFIROrderedAssignments : Pass<"lower-hlfir-ordered-assignments", "::mlir::ModuleOp"> {
  let summary = "Lower HLFIR ordered assignments like forall and where operations";
  let options = [
    Option<"tryFusingAssignments", "fuse-assignments",
           "bool", /*default=*/"false",
           "Fuse Forall and Where assignments in the same loop nest when legal."
           "It is not clear yet if this is always beneficial. It may be best to"
           "leave this to later loop optimizations."
           "Hence this is off by default.">
  ];
}

def SimplifyHLFIRIntrinsics : Pass<"simplify-hlfir-intrinsics"> {
  let summary = "Simplify HLFIR intrinsic operations that don't need to result in runtime calls";
  let options = [Option<"allowNewSideEffects", "allow-new-side-effects", "bool",
                        /*default=*/"false",
                        "If enabled, then the HLFIR operations simplification "
                        "may introduce operations with side effects. "
                        "For example, hlfir.matmul may be inlined as "
                        "and hlfir.eval_in_mem with hlfir.assign inside it."
                        "The hlfir.assign has a write effect on the memory "
                        "argument of hlfir.eval_in_mem, which may block "
                        "some existing MLIR transformations (e.g. CSE) "
                        "that otherwise would have been possible across "
                        "the hlfir.matmul.">,
                 Option<"fpMaxminBehavior", "fp-maxmin-behavior",
                        "Fortran::common::FPMaxminBehavior",
                        /*default=*/"Fortran::common::FPMaxminBehavior::Legacy",
                        "This option defines the FP max/min behavior.",
                        [{::llvm::cl::values(
               clEnumValN(Fortran::common::FPMaxminBehavior::Legacy,
                          "legacy", "cmp+select"),
               clEnumValN(Fortran::common::FPMaxminBehavior::Portable,
                          "portable", "same as legacy, except that "
                          "arith.max/minnumf may be used when nnan "
                          "and nsz fast math flags are enabled"),
               clEnumValN(Fortran::common::FPMaxminBehavior::Extremum,
                          "extremum", "arith.max/minimumf"),
               clEnumValN(Fortran::common::FPMaxminBehavior::ExtremeNum,
                          "extremenum", "arith.max/minnumf"))}]>];
}

def ExpressionSimplification : Pass<"hlfir-expression-simplification"> {
  let summary = "Simplify Fortran expressions";
}

def InlineElementals : Pass<"inline-elementals"> {
  let summary = "Inline chained hlfir.elemental operations";
}

def SeparateAllocatableAssign : Pass<"separate-allocatable-assign"> {
  let summary = "Separate reallocation from allocatable array assignments";
  let description = [{
    Transform `hlfir.assign %rhs to %lhs realloc` into a conditional
    reallocation of the LHS followed by a non-realloc `hlfir.assign`.
    This separates host-side allocation from device-side computation
    for OpenACC/OpenMP offloading and runs at all optimization levels.
  }];
}

def InlineHLFIRAssign : Pass<"inline-hlfir-assign"> {
  let summary = "Inline hlfir.assign operations";
  let options = [Option<
      "onlyScalarRHS", "only-scalar-rhs", "bool", /*default=*/"false",
      "Only inline assignments where RHS is a numeric or logical scalar "
      "broadcast "
      "to an array LHS. Used at O0 to avoid runtime calls in "
      "device code without inlining all array assignments.">];
}

def InlineHLFIRCopy : Pass<"inline-hlfir-copy"> {
  let summary = "Inline hlfir.copy_in and hlfir.copy_out operations";
}

def PropagateFortranVariableAttributes : Pass<"propagate-fortran-attrs"> {
  let summary = "Propagate FortranVariableFlagsAttr attributes through HLFIR";
}

def ArraySectionReduction : Pass<"array-section-reduction"> {
  let summary = "Promote loop-invariant array-section reductions to a local temporary";
  let description = [{
    Detect a loop-carried reduction into a loop-invariant array section that is
    unconditionally defined before the loop (e.g. `a(:,i) = a(:,i) + x(:)`), and
    promote the section to a constant-shape local temporary that is written back
    after the loop. The loop is then annotated to enable vectorization: promoting
    the section removes the memory dependence on the loop-invariant descriptor
    that would otherwise block the loop vectorizer.
  }];
  let options = [
      Option<"forceVectorize", "force-vectorize", "bool", /*default=*/"true",
             "Annotate the promoted loop with llvm.loop.vectorize.enable to "
             "override the cost model. The promoted temporary has a "
             "compile-time constant shape, so it scalarizes into individual "
             "registers and the loop becomes a plain reduction; disable to "
             "leave the vectorization decision to the cost model.">,
      Option<"maxReductionExtent", "max-reduction-extent", "unsigned",
             /*default=*/"8",
             "Only promote sections whose constant element count is at most "
             "this. Each element becomes a reduction accumulator kept in a "
             "vector register across the loop, so a too-large count exhausts "
             "the registers and spills.">,
      Option<"maxUnrollableTripCount", "max-unrollable-trip-count", "unsigned",
             /*default=*/"64",
             "A constant-bound inner loop with at most this trip count is "
             "assumed to be fully unrolled before the vectorizer runs, so it "
             "does not block annotating the enclosing reduction loop. This "
             "only affects whether the vectorization hint is emitted, never "
             "correctness.">];
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif //FORTRAN_DIALECT_HLFIR_PASSES

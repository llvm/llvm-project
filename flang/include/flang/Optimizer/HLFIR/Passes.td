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
  let constructor = "hlfir::createConvertHLFIRtoFIRPass()";
}

def BufferizeHLFIR : Pass<"bufferize-hlfir", "::mlir::ModuleOp"> {
  let summary = "Convert HLFIR operations operating on hlfir.expr into operations on memory";
  let constructor = "hlfir::createBufferizeHLFIRPass()";
}

def OptimizedBufferization : Pass<"opt-bufferization", "::mlir::func::FuncOp"> {
  let summary = "Special cases for hlfir.expr bufferization where we can avoid a temporary which would be created by the generic bufferization pass";
  let constructor = "hlfir::createOptimizedBufferizationPass()";
}

def LowerHLFIRIntrinsics : Pass<"lower-hlfir-intrinsics", "::mlir::ModuleOp"> {
  let summary = "Lower HLFIR transformational intrinsic operations";
  let constructor = "hlfir::createLowerHLFIRIntrinsicsPass()";
}

def LowerHLFIROrderedAssignments : Pass<"lower-hlfir-ordered-assignments", "::mlir::ModuleOp"> {
  let summary = "Lower HLFIR ordered assignments like forall and where operations";
  let constructor = "hlfir::createLowerHLFIROrderedAssignmentsPass()";
  let options = [
    Option<"tryFusingAssignments", "fuse-assignments",
           "bool", /*default=*/"false",
           "Fuse Forall and Where assignments in the same loop nest when legal."
           "It is not clear yet if this is always beneficial. It may be best to"
           "leave this to later loop optimizations."
           "Hence this is off by default.">
  ];
}

def SimplifyHLFIRIntrinsics : Pass<"simplify-hlfir-intrinsics", "::mlir::func::FuncOp"> {
  let summary = "Simplify HLFIR intrinsic operations that don't need to result in runtime calls";
  let constructor = "hlfir::createSimplifyHLFIRIntrinsicsPass()";
}

def InlineElementals : Pass<"inline-elementals", "::mlir::func::FuncOp"> {
  let summary = "Inline chained hlfir.elemental operations";
  let constructor = "hlfir::createInlineElementalsPass()";
}

#endif //FORTRAN_DIALECT_HLFIR_PASSES

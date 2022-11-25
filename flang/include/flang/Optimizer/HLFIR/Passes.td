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
def ConvertHLFIRtoFIR : Pass<"convert-hlfir-to-fir", "::mlir::func::FuncOp"> {
  let summary = "Lower High-Level FIR to FIR";
  let constructor = "hlfir::createConvertHLFIRtoFIRPass()";
}

def BufferizeHLFIR : Pass<"bufferize-hlfir", "::mlir::ModuleOp"> {
  let summary = "Convert HLFIR operations operating on hlfir.expr into operations on memory";
  let constructor = "hlfir::createBufferizeHLFIRPass()";
}

#endif //FORTRAN_DIALECT_HLFIR_PASSES

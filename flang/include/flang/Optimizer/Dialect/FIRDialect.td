//===-- FIRDialect.td - FIR dialect definition -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the FIR dialect
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_FIR_DIALECT
#define FORTRAN_DIALECT_FIR_DIALECT

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def fir_Dialect : Dialect {
  let name = "fir";
  let cppNamespace = "::fir";
  let emitAccessorPrefix = kEmitAccessorPrefix_Both;
}

#endif // FORTRAN_DIALECT_FIR_DIALECT

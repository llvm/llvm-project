//===- CIROpInterfaces.td - CIR Op Interface Definitions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_OP_INTERFACES
#define MLIR_CIR_OP_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"

let cppNamespace = "::mlir::cir" in {
  // The CIRCallOpInterface must be used instead of CallOpInterface when looking
  // at arguments and other bits of CallOp. This creates a level of abstraction
  // that's useful for handling indirect calls and other details.
  def CIRCallOpInterface : OpInterface<"CIRCallOpInterface", [CallOpInterface]> {
    let methods = [
      InterfaceMethod<"", "mlir::Operation::operand_iterator",
                      "arg_operand_begin", (ins)>,
      InterfaceMethod<"", "mlir::Operation::operand_iterator",
                      "arg_operand_end", (ins)>,
      InterfaceMethod<
        "Return the operand at index 'i', accounts for indirect call or "
        "exception info", "mlir::Value", "getArgOperand", (ins "unsigned":$i)>,
      InterfaceMethod<
        "Return the number of operands, accounts for indirect call or "
        "exception info", "unsigned", "getNumArgOperands", (ins)>,
    ];
  }

} // namespace mlir::cir

#endif // MLIR_CIR_OP_INTERFACES

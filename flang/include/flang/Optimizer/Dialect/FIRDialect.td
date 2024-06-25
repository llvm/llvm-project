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

def FIROpsDialect : Dialect {
  let name = "fir";
  let cppNamespace = "::fir";
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;
  let usePropertiesForAttributes = 1;
  let dependentDialects = [
    // Arith dialect provides FastMathFlagsAttr
    // supported by some FIR operations.
    "mlir::arith::ArithDialect",
    // TBAA Tag types
    "mlir::LLVM::LLVMDialect"
  ];
  let extraClassDeclaration = [{
  private:
    // Register the builtin Attributes.
    void registerAttributes();
    // Register the builtin Types.
    void registerTypes();
    // Register external interfaces on operations of
    // this dialect.
    void registerOpExternalInterfaces();
  public:
    mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
    void printType(mlir::Type ty, mlir::DialectAsmPrinter &p) const override;
 
    mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                   mlir::Type type) const override;
    void printAttribute(mlir::Attribute attr,
                        mlir::DialectAsmPrinter &p) const override;

    // Return string name of fir.runtime attribute.
    static constexpr llvm::StringRef getFirRuntimeAttrName() {
      return "fir.runtime";
    }
  }];
}

#endif // FORTRAN_DIALECT_FIR_DIALECT

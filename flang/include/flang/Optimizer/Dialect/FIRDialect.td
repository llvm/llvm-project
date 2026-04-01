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

include "aiir/IR/SymbolInterfaces.td"
include "aiir/Interfaces/CallInterfaces.td"
include "aiir/Interfaces/ControlFlowInterfaces.td"
include "aiir/Interfaces/InferTypeOpInterface.td"
include "aiir/Interfaces/LoopLikeInterface.td"
include "aiir/Interfaces/SideEffectInterfaces.td"

def FIROpsDialect : Dialect {
  let name = "fir";
  let cppNamespace = "::fir";
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;
  let dependentDialects = [
    // Arith dialect provides FastMathFlagsAttr
    // supported by some FIR operations.
    "aiir::arith::ArithDialect",
    // TBAA Tag types
    "aiir::LLVM::LLVMDialect"
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
    aiir::Type parseType(aiir::DialectAsmParser &parser) const override;
    void printType(aiir::Type ty, aiir::DialectAsmPrinter &p) const override;
 
    aiir::Attribute parseAttribute(aiir::DialectAsmParser &parser,
                                   aiir::Type type) const override;
    void printAttribute(aiir::Attribute attr,
                        aiir::DialectAsmPrinter &p) const override;

    // Return string name of fir.runtime attribute.
    static constexpr llvm::StringRef getFirRuntimeAttrName() {
      return "fir.runtime";
    }
    // Return string name of fir.memory attributes.
    // It is attached to fir.call operations to convey
    // llvm.memory attributes to LLVM IR.
    // Its value is intended to be aiir::LLVM::MemoryEffectsAttr.
    // TODO: we should probably make it an inherent attribute
    // of fir.call, though, it is supposed to be a short-lived
    // attribute that appears right before CodeGen and only
    // meaningful for LLVM, so it is unclear if embedding
    // it into fir.call makes sense.
    static constexpr llvm::StringRef getFirCallMemoryAttrName() {
      return "fir.llvm_memory";
    }
  }];
}

#endif // FORTRAN_DIALECT_FIR_DIALECT

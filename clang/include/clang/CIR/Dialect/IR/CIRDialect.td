//===- CIRDialect.td - CIR dialect -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRDIALECT_TD
#define CLANG_CIR_DIALECT_IR_CIRDIALECT_TD

include "mlir/IR/OpBase.td"

def CIR_Dialect : Dialect {
  let name = "cir";

  // A short one-line summary of our dialect.
  let summary = "A high-level dialect for analyzing and optimizing Clang "
                "supported languages";

  let cppNamespace = "::cir";

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;

  // Enable constant materialization for the CIR dialect. This generates a
  // declaration for the cir::CIRDialect::materializeConstant function. This
  // hook is necessary for canonicalization to properly handle attributes
  // returned by fold methods, allowing them to be materialized as constant
  // operations in the IR.
  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
    static llvm::StringRef getSourceLanguageAttrName() { return "cir.lang"; }
    static llvm::StringRef getTripleAttrName() { return "cir.triple"; }
    static llvm::StringRef getOptInfoAttrName() { return "cir.opt_info"; }
    static llvm::StringRef getCalleeAttrName() { return "callee"; }
    static llvm::StringRef getNoThrowAttrName() { return "nothrow"; }
    static llvm::StringRef getSideEffectAttrName() { return "side_effect"; }
    static llvm::StringRef getModuleLevelAsmAttrName() { return "cir.module_asm"; }
    static llvm::StringRef getGlobalCtorsAttrName() { return "cir.global_ctors"; }
    static llvm::StringRef getGlobalDtorsAttrName() { return "cir.global_dtors"; }

    void registerAttributes();
    void registerTypes();

    mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
    void printType(mlir::Type type,
                   mlir::DialectAsmPrinter &printer) const override;

    mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                   mlir::Type type) const override;

    void printAttribute(mlir::Attribute attr,
                        mlir::DialectAsmPrinter &os) const override;
  }];
}

#endif // CLANG_CIR_DIALECT_IR_CIRDIALECT_TD

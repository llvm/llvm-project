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

#ifndef LLVM_CLANG_CIR_DIALECT_IR_CIRDIALECT
#define LLVM_CLANG_CIR_DIALECT_IR_CIRDIALECT

include "mlir/IR/OpBase.td"

def CIR_Dialect : Dialect {
  let name = "cir";

  // A short one-line summary of our dialect.
  let summary = "A high-level dialect for analyzing and optimizing Clang "
                "supported languages";

  let cppNamespace = "::cir";

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;

  let extraClassDeclaration = [{
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

#endif // LLVM_CLANG_CIR_DIALECT_IR_CIRDIALECT

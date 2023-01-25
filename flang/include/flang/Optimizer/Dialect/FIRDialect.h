//===-- Optimizer/Dialect/FIRDialect.h -- FIR dialect -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_DIALECT_FIRDIALECT_H
#define FORTRAN_OPTIMIZER_DIALECT_FIRDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
class IRMapping;
} // namespace mlir

namespace fir {

/// FIR dialect
class FIROpsDialect final : public mlir::Dialect {
public:
  explicit FIROpsDialect(mlir::MLIRContext *ctx);
  virtual ~FIROpsDialect();

  static llvm::StringRef getDialectNamespace() { return "fir"; }

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type ty, mlir::DialectAsmPrinter &p) const override;

  mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                 mlir::Type type) const override;
  void printAttribute(mlir::Attribute attr,
                      mlir::DialectAsmPrinter &p) const override;

private:
  // Register the Attributes of this dialect.
  void registerAttributes();
  // Register the Types of this dialect.
  void registerTypes();
};

/// The FIR codegen dialect is a dialect containing a small set of transient
/// operations used exclusively during code generation.
class FIRCodeGenDialect final : public mlir::Dialect {
public:
  explicit FIRCodeGenDialect(mlir::MLIRContext *ctx);
  virtual ~FIRCodeGenDialect();

  static llvm::StringRef getDialectNamespace() { return "fircg"; }
};

/// Support for inlining on FIR.
bool canLegallyInline(mlir::Operation *op, mlir::Region *reg, bool,
                      mlir::IRMapping &map);
bool canLegallyInline(mlir::Operation *, mlir::Operation *, bool);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_DIALECT_FIRDIALECT_H

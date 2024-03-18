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

#include "flang/Optimizer/Dialect/FIRDialect.h.inc"

namespace mlir {
class IRMapping;
} // namespace mlir

namespace fir {

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

// Register the FIRInlinerInterface to FIROpsDialect
void addFIRInlinerExtension(mlir::DialectRegistry &registry);

// Register implementation of LLVMTranslationDialectInterface.
void addFIRToLLVMIRExtension(mlir::DialectRegistry &registry);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_DIALECT_FIRDIALECT_H

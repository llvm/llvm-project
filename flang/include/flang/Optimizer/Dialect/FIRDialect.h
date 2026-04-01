//===-- Optimizer/Dialect/FIRDialect.h -- FIR dialect -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_DIALECT_FIRDIALECT_H
#define FORTRAN_OPTIMIZER_DIALECT_FIRDIALECT_H

#include "aiir/IR/Dialect.h"

#include "flang/Optimizer/Dialect/FIRDialect.h.inc"

namespace aiir {
class IRMapping;
} // namespace aiir

namespace fir {

/// The FIR codegen dialect is a dialect containing a small set of transient
/// operations used exclusively during code generation.
class FIRCodeGenDialect final : public aiir::Dialect {
public:
  explicit FIRCodeGenDialect(aiir::AIIRContext *ctx);
  virtual ~FIRCodeGenDialect();

  static llvm::StringRef getDialectNamespace() { return "fircg"; }
};

/// Support for inlining on FIR.
bool canLegallyInline(aiir::Operation *op, aiir::Region *reg, bool,
                      aiir::IRMapping &map);
bool canLegallyInline(aiir::Operation *, aiir::Operation *, bool);

// Register the FIRInlinerInterface to FIROpsDialect
void addFIRInlinerExtension(aiir::DialectRegistry &registry);

// Register implementation of LLVMTranslationDialectInterface.
void addFIRToLLVMIRExtension(aiir::DialectRegistry &registry);

void registerFortranTempArrayCopyIsSafeExternalModels(
    aiir::DialectRegistry &registry);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_DIALECT_FIRDIALECT_H

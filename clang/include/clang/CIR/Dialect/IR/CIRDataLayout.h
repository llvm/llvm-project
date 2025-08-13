//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Provides an LLVM-like API wrapper to DLTI and MLIR layout queries. This
// makes it easier to port some of LLVM codegen layout logic to CIR.
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRDATALAYOUT_H
#define CLANG_CIR_DIALECT_IR_CIRDATALAYOUT_H

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinOps.h"

namespace cir {

// TODO(cir): This might be replaced by a CIRDataLayout interface which can
// provide the same functionalities.
class CIRDataLayout {
  // This is starting with the minimum functionality needed for code that is
  // being upstreamed. Additional methods and members will be added as needed.
  bool bigEndian = false;

public:
  mlir::DataLayout layout;

  /// Constructs a DataLayout the module's data layout attribute.
  CIRDataLayout(mlir::ModuleOp modOp);

  /// Parse a data layout string (with fallback to default values).
  void reset(mlir::DataLayoutSpecInterface spec);

  bool isBigEndian() const { return bigEndian; }

  /// Returns the maximum number of bytes that may be overwritten by
  /// storing the specified type.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// For example, returns 5 for i36 and 10 for x86_fp80.
  llvm::TypeSize getTypeStoreSize(mlir::Type ty) const {
    llvm::TypeSize baseSize = getTypeSizeInBits(ty);
    return {llvm::divideCeil(baseSize.getKnownMinValue(), 8),
            baseSize.isScalable()};
  }

  llvm::TypeSize getTypeSizeInBits(mlir::Type ty) const;
};

} // namespace cir

#endif // CLANG_CIR_DIALECT_IR_CIRDATALAYOUT_H

//===--- CIRDataLayout.h - CIR Data Layout Information ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Provides a LLVM-like API wrapper to DLTI and MLIR layout queries. This makes
// it easier to port some of LLVM codegen layout logic to CIR.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_IR_CIRDATALAYOUT_H
#define LLVM_CLANG_CIR_DIALECT_IR_CIRDATALAYOUT_H

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/StringRef.h"

namespace cir {

class CIRDataLayout {
  bool bigEndian = false;

public:
  mlir::DataLayout layout;

  /// Constructs a DataLayout from a specification string. See reset().
  explicit CIRDataLayout(llvm::StringRef dataLayout, mlir::ModuleOp module)
      : layout(module) {
    reset(dataLayout);
  }

  /// Parse a data layout string (with fallback to default values).
  void reset(llvm::StringRef dataLayout);

  // Free all internal data structures.
  void clear();

  CIRDataLayout(mlir::ModuleOp modOp);
  bool isBigEndian() const { return bigEndian; }

  // `useABI` is `true` if not using prefered alignment.
  unsigned getAlignment(mlir::Type ty, bool useABI) const {
    if (llvm::isa<mlir::cir::StructType>(ty)) {
      auto sTy = ty.cast<mlir::cir::StructType>();
      if (sTy.getPacked() && useABI)
        return 1;
    } else if (llvm::isa<mlir::cir::ArrayType>(ty)) {
      return getAlignment(ty.cast<mlir::cir::ArrayType>().getEltType(), useABI);
    }

    return useABI ? layout.getTypeABIAlignment(ty)
                  : layout.getTypePreferredAlignment(ty);
  }

  unsigned getABITypeAlign(mlir::Type ty) const {
    return getAlignment(ty, true);
  }

  /// Returns the maximum number of bytes that may be overwritten by
  /// storing the specified type.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// For example, returns 5 for i36 and 10 for x86_fp80.
  unsigned getTypeStoreSize(mlir::Type Ty) const {
    // FIXME: this is a bit inaccurate, see DataLayout::getTypeStoreSize for
    // more information.
    return llvm::divideCeil(layout.getTypeSizeInBits(Ty), 8);
  }

  /// Returns the offset in bytes between successive objects of the
  /// specified type, including alignment padding.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// This is the amount that alloca reserves for this type. For example,
  /// returns 12 or 16 for x86_fp80, depending on alignment.
  unsigned getTypeAllocSize(mlir::Type Ty) const {
    // Round up to the next alignment boundary.
    return llvm::alignTo(getTypeStoreSize(Ty), getABITypeAlign(Ty));
  }

  unsigned getPointerTypeSizeInBits(mlir::Type Ty) const {
    assert(Ty.isa<mlir::cir::PointerType>() &&
           "This should only be called with a pointer type");
    return layout.getTypeSizeInBits(Ty);
  }

  unsigned getTypeSizeInBits(mlir::Type Ty) const {
    return layout.getTypeSizeInBits(Ty);
  }

  mlir::Type getIntPtrType(mlir::Type Ty) const {
    assert(Ty.isa<mlir::cir::PointerType>() && "Expected pointer type");
    auto IntTy = mlir::cir::IntType::get(Ty.getContext(),
                                         getPointerTypeSizeInBits(Ty), false);
    return IntTy;
  }
};

} // namespace cir

#endif

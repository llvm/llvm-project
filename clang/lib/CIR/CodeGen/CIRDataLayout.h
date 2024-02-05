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

#ifndef LLVM_CLANG_LIB_CIR_CIRDATALAYOUT_H
#define LLVM_CLANG_LIB_CIR_CIRDATALAYOUT_H

#include "UnimplementedFeatureGuarding.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace cir {

class CIRDataLayout {
  bool bigEndian = false;

public:
  mlir::DataLayout layout;

  CIRDataLayout(mlir::ModuleOp modOp);
  bool isBigEndian() const { return bigEndian; }

  // `useABI` is `true` if not using prefered alignment.
  unsigned getAlignment(mlir::Type ty, bool useABI) const {
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
    return llvm::alignTo(getTypeStoreSize(Ty), layout.getTypeABIAlignment(Ty));
  }

  unsigned getPointerTypeSizeInBits(mlir::Type Ty) const {
    assert(Ty.isa<mlir::cir::PointerType>() &&
           "This should only be called with a pointer type");
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
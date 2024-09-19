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
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/TypeSize.h"

namespace cir {

class StructLayout;

// FIXME(cir): This might be replaced by a CIRDataLayout interface which can
// provide the same functionalities.
class CIRDataLayout {
  bool bigEndian = false;

  /// Primitive type alignment data. This is sorted by type and bit
  /// width during construction.
  llvm::LayoutAlignElem StructAlignment;

  // The StructType -> StructLayout map.
  mutable void *LayoutMap = nullptr;

public:
  mlir::DataLayout layout;

  /// Constructs a DataLayout the module's data layout attribute.
  CIRDataLayout(mlir::ModuleOp modOp);

  /// Parse a data layout string (with fallback to default values).
  void reset(mlir::DataLayoutSpecInterface spec);

  // Free all internal data structures.
  void clear();

  bool isBigEndian() const { return bigEndian; }

  /// Returns a StructLayout object, indicating the alignment of the
  /// struct, its size, and the offsets of its fields.
  ///
  /// Note that this information is lazily cached.
  const StructLayout *getStructLayout(mlir::cir::StructType Ty) const;

  /// Internal helper method that returns requested alignment for type.
  llvm::Align getAlignment(mlir::Type Ty, bool abiOrPref) const;

  llvm::Align getABITypeAlign(mlir::Type ty) const {
    return getAlignment(ty, true);
  }

  /// Returns the maximum number of bytes that may be overwritten by
  /// storing the specified type.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// For example, returns 5 for i36 and 10 for x86_fp80.
  llvm::TypeSize getTypeStoreSize(mlir::Type Ty) const {
    llvm::TypeSize BaseSize = getTypeSizeInBits(Ty);
    return {llvm::divideCeil(BaseSize.getKnownMinValue(), 8),
            BaseSize.isScalable()};
  }

  /// Returns the offset in bytes between successive objects of the
  /// specified type, including alignment padding.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// This is the amount that alloca reserves for this type. For example,
  /// returns 12 or 16 for x86_fp80, depending on alignment.
  llvm::TypeSize getTypeAllocSize(mlir::Type Ty) const {
    // Round up to the next alignment boundary.
    return llvm::alignTo(getTypeStoreSize(Ty), getABITypeAlign(Ty).value());
  }

  llvm::TypeSize getPointerTypeSizeInBits(mlir::Type Ty) const {
    assert(mlir::isa<mlir::cir::PointerType>(Ty) &&
           "This should only be called with a pointer type");
    return layout.getTypeSizeInBits(Ty);
  }

  llvm::TypeSize getTypeSizeInBits(mlir::Type Ty) const;

  mlir::Type getIntPtrType(mlir::Type Ty) const {
    assert(mlir::isa<mlir::cir::PointerType>(Ty) && "Expected pointer type");
    auto IntTy = mlir::cir::IntType::get(Ty.getContext(),
                                         getPointerTypeSizeInBits(Ty), false);
    return IntTy;
  }
};

/// Used to lazily calculate structure layout information for a target machine,
/// based on the DataLayout structure.
class StructLayout final
    : public llvm::TrailingObjects<StructLayout, llvm::TypeSize> {
  llvm::TypeSize StructSize;
  llvm::Align StructAlignment;
  unsigned IsPadded : 1;
  unsigned NumElements : 31;

public:
  llvm::TypeSize getSizeInBytes() const { return StructSize; }

  llvm::TypeSize getSizeInBits() const { return 8 * StructSize; }

  llvm::Align getAlignment() const { return StructAlignment; }

  /// Returns whether the struct has padding or not between its fields.
  /// NB: Padding in nested element is not taken into account.
  bool hasPadding() const { return IsPadded; }

  /// Given a valid byte offset into the structure, returns the structure
  /// index that contains it.
  unsigned getElementContainingOffset(uint64_t FixedOffset) const;

  llvm::MutableArrayRef<llvm::TypeSize> getMemberOffsets() {
    return llvm::MutableArrayRef(getTrailingObjects<llvm::TypeSize>(),
                                 NumElements);
  }

  llvm::ArrayRef<llvm::TypeSize> getMemberOffsets() const {
    return llvm::ArrayRef(getTrailingObjects<llvm::TypeSize>(), NumElements);
  }

  llvm::TypeSize getElementOffset(unsigned Idx) const {
    assert(Idx < NumElements && "Invalid element idx!");
    return getMemberOffsets()[Idx];
  }

  llvm::TypeSize getElementOffsetInBits(unsigned Idx) const {
    return getElementOffset(Idx) * 8;
  }

private:
  friend class CIRDataLayout; // Only DataLayout can create this class

  StructLayout(mlir::cir::StructType ST, const CIRDataLayout &DL);

  size_t numTrailingObjects(OverloadToken<llvm::TypeSize>) const {
    return NumElements;
  }
};

} // namespace cir

#endif

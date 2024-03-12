//===-- Address.h - An aligned address -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class provides a simple wrapper for a pair of a pointer and an
// alignment.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_ADDRESS_H
#define LLVM_CLANG_LIB_CIR_ADDRESS_H

#include "clang/AST/CharUnits.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "llvm/IR/Constants.h"

#include "mlir/IR/Value.h"

namespace cir {

// Indicates whether a pointer is known not to be null.
enum KnownNonNull_t { NotKnownNonNull, KnownNonNull };

class Address {
  llvm::PointerIntPair<mlir::Value, 1, bool> PointerAndKnownNonNull;
  mlir::Type ElementType;
  clang::CharUnits Alignment;

protected:
  Address(std::nullptr_t) : ElementType(nullptr) {}

public:
  Address(mlir::Value pointer, mlir::Type elementType,
          clang::CharUnits alignment,
          KnownNonNull_t IsKnownNonNull = NotKnownNonNull)
      : PointerAndKnownNonNull(pointer, IsKnownNonNull),
        ElementType(elementType), Alignment(alignment) {
    assert(pointer.getType().isa<mlir::cir::PointerType>() &&
           "Expected cir.ptr type");

    assert(pointer && "Pointer cannot be null");
    assert(elementType && "Element type cannot be null");
    assert(!alignment.isZero() && "Alignment cannot be zero");
  }
  Address(mlir::Value pointer, clang::CharUnits alignment)
      : Address(pointer,
                pointer.getType().cast<mlir::cir::PointerType>().getPointee(),
                alignment) {

    assert((!alignment.isZero() || pointer == nullptr) &&
           "creating valid address with invalid alignment");
  }

  static Address invalid() { return Address(nullptr); }
  bool isValid() const {
    return PointerAndKnownNonNull.getPointer() != nullptr;
  }

  /// Return address with different pointer, but same element type and
  /// alignment.
  Address withPointer(mlir::Value NewPointer,
                      KnownNonNull_t IsKnownNonNull) const {
    return Address(NewPointer, getElementType(), getAlignment(),
                   IsKnownNonNull);
  }

  /// Return address with different alignment, but same pointer and element
  /// type.
  Address withAlignment(clang::CharUnits NewAlignment) const {
    return Address(getPointer(), getElementType(), NewAlignment,
                   isKnownNonNull());
  }

  /// Return address with different element type, but same pointer and
  /// alignment.
  Address withElementType(mlir::Type ElemTy) const {
    return Address(getPointer(), ElemTy, getAlignment(), isKnownNonNull());
  }

  mlir::Value getPointer() const {
    assert(isValid());
    return PointerAndKnownNonNull.getPointer();
  }

  /// Return the alignment of this pointer.
  clang::CharUnits getAlignment() const {
    // assert(isValid());
    return Alignment;
  }

  /// Return the type of the pointer value.
  mlir::cir::PointerType getType() const {
    return getPointer().getType().cast<mlir::cir::PointerType>();
  }

  mlir::Type getElementType() const {
    assert(isValid());
    return ElementType;
  }

  /// Whether the pointer is known not to be null.
  KnownNonNull_t isKnownNonNull() const {
    assert(isValid());
    return (KnownNonNull_t)PointerAndKnownNonNull.getInt();
  }

  /// Set the non-null bit.
  Address setKnownNonNull() {
    assert(isValid());
    PointerAndKnownNonNull.setInt(true);
    return *this;
  }

  /// Get the operation which defines this address.
  mlir::Operation *getDefiningOp() const {
    if (!isValid())
      return nullptr;
    return getPointer().getDefiningOp();
  }
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_ADDRESS_H

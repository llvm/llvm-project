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

#include "CIRGenPointerAuthInfo.h"

#include "mlir/IR/Value.h"

namespace clang::CIRGen {

// Forward declaration to avoid a circular dependency
class CIRGenBuilderTy;

// Indicates whether a pointer is known not to be null.
enum KnownNonNull_t { NotKnownNonNull, KnownNonNull };

/// Like RawAddress, an abstract representation of an aligned address, but the
/// pointer contained in this class is possibly signed.
class Address {

  // The boolean flag indicates whether the pointer is known to be non-null.
  llvm::PointerIntPair<mlir::Value, 1, bool> PointerAndKnownNonNull;

  /// The expected CIR type of the pointer. Carrying accurate element type
  /// information in Address makes it more convenient to work with Address
  /// values and allows frontend assertions to catch simple mistakes.
  mlir::Type ElementType;

  clang::CharUnits Alignment;

  /// The ptrauth information needed to authenticate the base pointer.
  cir::CIRGenPointerAuthInfo ptrAuthInfo;

  /// Offset from the base pointer. This is non-null only when the base pointer
  /// is signed.
  mlir::Value offset = nullptr;

protected:
  Address(std::nullptr_t) : ElementType(nullptr) {}

public:
  Address(mlir::Value pointer, mlir::Type elementType,
          clang::CharUnits alignment,
          KnownNonNull_t IsKnownNonNull = NotKnownNonNull)
      : PointerAndKnownNonNull(pointer, IsKnownNonNull),
        ElementType(elementType), Alignment(alignment) {
    assert(mlir::isa<cir::PointerType>(pointer.getType()) &&
           "Expected cir.ptr type");

    assert(pointer && "Pointer cannot be null");
    assert(elementType && "Element type cannot be null");
    assert(!alignment.isZero() && "Alignment cannot be zero");

    assert(mlir::cast<cir::PointerType>(pointer.getType()).getPointee() ==
           ElementType);
  }

  Address(mlir::Value basePtr, mlir::Type elementType,
          clang::CharUnits alignment, cir::CIRGenPointerAuthInfo ptrAuthInfo,
          mlir::Value offset, KnownNonNull_t isKnownNonNull = NotKnownNonNull)
      : PointerAndKnownNonNull(basePtr, isKnownNonNull),
        ElementType(elementType), Alignment(alignment),
        ptrAuthInfo(ptrAuthInfo), offset(offset) {}

  Address(mlir::Value pointer, clang::CharUnits alignment)
      : Address(pointer,
                mlir::cast<cir::PointerType>(pointer.getType()).getPointee(),
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
                      KnownNonNull_t IsKnownNonNull = NotKnownNonNull) const {
    return Address(NewPointer, getElementType(), getAlignment(),
                   IsKnownNonNull);
  }

  /// Return address with different alignment, but same pointer and element
  /// type.
  Address withAlignment(clang::CharUnits NewAlignment) const {
    return Address(getPointer(), getElementType(), NewAlignment,
                   isKnownNonNull());
  }

  bool hasOffset() const { return bool(offset); }

  /// Return address with different element type, a bitcast pointer, and
  /// the same alignment.
  Address withElementType(CIRGenBuilderTy &builder, mlir::Type ElemTy) const;

  mlir::Value getPointer() const {
    assert(isValid());
    return PointerAndKnownNonNull.getPointer();
  }

  mlir::Value getBasePointer() const {
    // TODO(cir): Remove the version above when we catchup with OG codegen on
    // ptr auth.
    assert(isValid() && "pointer isn't valid");
    return getPointer();
  }

  /// Return the alignment of this pointer.
  clang::CharUnits getAlignment() const {
    // assert(isValid());
    return Alignment;
  }

  /// Return the pointer contained in this class after authenticating it and
  /// adding offset to it if necessary.
  mlir::Value emitRawPointer() const {
    // TODO(cir): update this class with latest traditional LLVM codegen bits
    // and the replace the call below to getBasePointer().
    return getPointer();
  }

  /// Return the type of the pointer value.
  cir::PointerType getType() const {
    assert(mlir::cast<cir::PointerType>(
               PointerAndKnownNonNull.getPointer().getType())
               .getPointee() == ElementType);
    return mlir::cast<cir::PointerType>(getPointer().getType());
  }

  mlir::Type getElementType() const {
    assert(isValid());
    assert(mlir::cast<cir::PointerType>(
               PointerAndKnownNonNull.getPointer().getType())
               .getPointee() == ElementType);
    return ElementType;
  }

  const cir::CIRGenPointerAuthInfo &getPointerAuthInfo() const {
    return ptrAuthInfo;
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

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_ADDRESS_H

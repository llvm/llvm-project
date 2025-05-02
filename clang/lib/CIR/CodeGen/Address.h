//===----------------------------------------------------------------------===//
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

#ifndef CLANG_LIB_CIR_ADDRESS_H
#define CLANG_LIB_CIR_ADDRESS_H

#include "mlir/IR/Value.h"
#include "clang/AST/CharUnits.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/PointerIntPair.h"

namespace clang::CIRGen {

class Address {

  // The boolean flag indicates whether the pointer is known to be non-null.
  llvm::PointerIntPair<mlir::Value, 1, bool> pointerAndKnownNonNull;

  /// The expected CIR type of the pointer. Carrying accurate element type
  /// information in Address makes it more convenient to work with Address
  /// values and allows frontend assertions to catch simple mistakes.
  mlir::Type elementType;

  clang::CharUnits alignment;

protected:
  Address(std::nullptr_t) : elementType(nullptr) {}

public:
  Address(mlir::Value pointer, mlir::Type elementType,
          clang::CharUnits alignment)
      : pointerAndKnownNonNull(pointer, false), elementType(elementType),
        alignment(alignment) {
    assert(mlir::isa<cir::PointerType>(pointer.getType()) &&
           "Expected cir.ptr type");

    assert(pointer && "Pointer cannot be null");
    assert(elementType && "Element type cannot be null");
    assert(!alignment.isZero() && "Alignment cannot be zero");

    assert(mlir::cast<cir::PointerType>(pointer.getType()).getPointee() ==
           elementType);
  }

  Address(mlir::Value pointer, clang::CharUnits alignment)
      : Address(pointer,
                mlir::cast<cir::PointerType>(pointer.getType()).getPointee(),
                alignment) {
    assert((!alignment.isZero() || pointer == nullptr) &&
           "creating valid address with invalid alignment");
  }

  static Address invalid() { return Address(nullptr); }
  bool isValid() const {
    return pointerAndKnownNonNull.getPointer() != nullptr;
  }

  mlir::Value getPointer() const {
    assert(isValid());
    return pointerAndKnownNonNull.getPointer();
  }

  mlir::Type getType() const {
    assert(mlir::cast<cir::PointerType>(
               pointerAndKnownNonNull.getPointer().getType())
               .getPointee() == elementType);

    return mlir::cast<cir::PointerType>(getPointer().getType());
  }

  mlir::Type getElementType() const {
    assert(isValid());
    assert(mlir::cast<cir::PointerType>(
               pointerAndKnownNonNull.getPointer().getType())
               .getPointee() == elementType);
    return elementType;
  }

  clang::CharUnits getAlignment() const { return alignment; }
};

} // namespace clang::CIRGen

#endif // CLANG_LIB_CIR_ADDRESS_H

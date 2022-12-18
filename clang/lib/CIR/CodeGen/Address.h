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
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace cir {

class Address {
  mlir::Value Pointer;
  mlir::Type ElementType;
  clang::CharUnits Alignment;

protected:
  Address(std::nullptr_t) : Pointer(nullptr), ElementType(nullptr) {}

public:
  Address(mlir::Value pointer, mlir::Type elementType,
          clang::CharUnits alignment)
      : Pointer(pointer), ElementType(elementType), Alignment(alignment) {
    auto ptrTy = pointer.getType().dyn_cast<mlir::cir::PointerType>();
    assert(ptrTy && "Expected cir.ptr type");

    assert(pointer != nullptr && "Pointer cannot be null");
    assert(elementType != nullptr && "Pointer cannot be null");
    assert(ptrTy.getPointee() == ElementType &&
           "Incorrect pointer element type");
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
  bool isValid() const { return Pointer != nullptr; }

  /// Return address with different pointer, but same element type and
  /// alignment.
  Address withPointer(mlir::Value NewPointer) const {
    return Address(NewPointer, getElementType(), getAlignment());
  }

  mlir::Value getPointer() const {
    // assert(isValid());
    return Pointer;
  }

  /// Return the alignment of this pointer.
  clang::CharUnits getAlignment() const {
    // assert(isValid());
    return Alignment;
  }

  mlir::Type getElementType() const {
    assert(isValid());
    return ElementType;
  }
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_ADDRESS_H

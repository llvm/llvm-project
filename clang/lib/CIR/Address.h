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

#include "llvm/IR/Constants.h"

#include "mlir/Dialect/CIR/IR/CIRTypes.h"
#include "mlir/IR/Value.h"

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
    assert(pointer != nullptr && "Pointer cannot be null");
    assert(elementType != nullptr && "Pointer cannot be null");
    assert(pointer.getType().cast<mlir::cir::PointerType>().getPointee() ==
               ElementType &&
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

/// A specialization of Address that requires the address to be an
/// MLIR attribute
class ConstantAddress : public Address {
  ConstantAddress(std::nullptr_t) : Address(nullptr) {}

public:
  ConstantAddress(mlir::Value pointer, mlir::Type elementType,
                  clang::CharUnits alignment)
      : Address(pointer, elementType, alignment) {}

  static ConstantAddress invalid() { return ConstantAddress(nullptr); }

  mlir::Value getPointer() const { return Address::getPointer(); }

  ConstantAddress getElementBitCast(mlir::Type ElemTy) const {
    assert(0 && "NYI");
  }

  static bool isaImpl(Address addr) {
    return addr.getPointer() ? true : false;
    // TODO(cir): in LLVM codegen this (and other methods) are implemented via
    // llvm::isa<llvm::Constant>, decide on what abstraction to use here.
    // return llvm::isa<llvm::Constant>(addr.getPointer());
  }
  static ConstantAddress castImpl(Address addr) {
    return ConstantAddress(addr.getPointer(), addr.getElementType(),
                           addr.getAlignment());
  }
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_ADDRESS_H

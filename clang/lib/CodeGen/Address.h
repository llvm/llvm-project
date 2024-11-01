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

#ifndef LLVM_CLANG_LIB_CODEGEN_ADDRESS_H
#define LLVM_CLANG_LIB_CODEGEN_ADDRESS_H

#include "clang/AST/CharUnits.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/MathExtras.h"

namespace clang {
namespace CodeGen {

// Indicates whether a pointer is known not to be null.
enum KnownNonNull_t { NotKnownNonNull, KnownNonNull };

/// An aligned address.
class Address {
  llvm::PointerIntPair<llvm::Value *, 1, bool> PointerAndKnownNonNull;
  llvm::Type *ElementType;
  CharUnits Alignment;

protected:
  Address(std::nullptr_t) : ElementType(nullptr) {}

public:
  Address(llvm::Value *Pointer, llvm::Type *ElementType, CharUnits Alignment,
          KnownNonNull_t IsKnownNonNull = NotKnownNonNull)
      : PointerAndKnownNonNull(Pointer, IsKnownNonNull),
        ElementType(ElementType), Alignment(Alignment) {
    assert(Pointer != nullptr && "Pointer cannot be null");
    assert(ElementType != nullptr && "Element type cannot be null");
  }

  static Address invalid() { return Address(nullptr); }
  bool isValid() const {
    return PointerAndKnownNonNull.getPointer() != nullptr;
  }

  llvm::Value *getPointer() const {
    assert(isValid());
    return PointerAndKnownNonNull.getPointer();
  }

  /// Return the type of the pointer value.
  llvm::PointerType *getType() const {
    return llvm::cast<llvm::PointerType>(getPointer()->getType());
  }

  /// Return the type of the values stored in this address.
  llvm::Type *getElementType() const {
    assert(isValid());
    return ElementType;
  }

  /// Return the address space that this address resides in.
  unsigned getAddressSpace() const {
    return getType()->getAddressSpace();
  }

  /// Return the IR name of the pointer value.
  llvm::StringRef getName() const {
    return getPointer()->getName();
  }

  /// Return the alignment of this pointer.
  CharUnits getAlignment() const {
    assert(isValid());
    return Alignment;
  }

  /// Return address with different pointer, but same element type and
  /// alignment.
  Address withPointer(llvm::Value *NewPointer,
                      KnownNonNull_t IsKnownNonNull) const {
    return Address(NewPointer, getElementType(), getAlignment(),
                   IsKnownNonNull);
  }

  /// Return address with different alignment, but same pointer and element
  /// type.
  Address withAlignment(CharUnits NewAlignment) const {
    return Address(getPointer(), getElementType(), NewAlignment,
                   isKnownNonNull());
  }

  /// Return address with different element type, but same pointer and
  /// alignment.
  Address withElementType(llvm::Type *ElemTy) const {
    return Address(getPointer(), ElemTy, getAlignment(), isKnownNonNull());
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
};

/// A specialization of Address that requires the address to be an
/// LLVM Constant.
class ConstantAddress : public Address {
  ConstantAddress(std::nullptr_t) : Address(nullptr) {}

public:
  ConstantAddress(llvm::Constant *pointer, llvm::Type *elementType,
                  CharUnits alignment)
      : Address(pointer, elementType, alignment) {}

  static ConstantAddress invalid() {
    return ConstantAddress(nullptr);
  }

  llvm::Constant *getPointer() const {
    return llvm::cast<llvm::Constant>(Address::getPointer());
  }

  ConstantAddress withElementType(llvm::Type *ElemTy) const {
    return ConstantAddress(getPointer(), ElemTy, getAlignment());
  }

  static bool isaImpl(Address addr) {
    return llvm::isa<llvm::Constant>(addr.getPointer());
  }
  static ConstantAddress castImpl(Address addr) {
    return ConstantAddress(llvm::cast<llvm::Constant>(addr.getPointer()),
                           addr.getElementType(), addr.getAlignment());
  }
};

}

// Present a minimal LLVM-like casting interface.
template <class U> inline U cast(CodeGen::Address addr) {
  return U::castImpl(addr);
}
template <class U> inline bool isa(CodeGen::Address addr) {
  return U::isaImpl(addr);
}

}

#endif

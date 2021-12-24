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

#include "llvm/IR/Constants.h"
#include "clang/AST/CharUnits.h"

namespace clang {
namespace CodeGen {

/// An aligned address.
class Address {
  llvm::Value *Pointer;
  llvm::Type *ElementType;
  CharUnits Alignment;

protected:
  Address(std::nullptr_t) : Pointer(nullptr), ElementType(nullptr) {}

public:
  Address(llvm::Value *pointer, llvm::Type *elementType, CharUnits alignment)
      : Pointer(pointer), ElementType(elementType), Alignment(alignment) {
    assert(pointer != nullptr && "Pointer cannot be null");
    assert(elementType != nullptr && "Element type cannot be null");
    assert(llvm::cast<llvm::PointerType>(pointer->getType())
               ->isOpaqueOrPointeeTypeMatches(elementType) &&
           "Incorrect pointer element type");
    assert(!alignment.isZero() && "Alignment cannot be zero");
  }

  // Deprecated: Use constructor with explicit element type instead.
  Address(llvm::Value *Pointer, CharUnits Alignment)
      : Address(Pointer, Pointer->getType()->getPointerElementType(),
                Alignment) {}

  static Address invalid() { return Address(nullptr); }
  bool isValid() const { return Pointer != nullptr; }

  llvm::Value *getPointer() const {
    assert(isValid());
    return Pointer;
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
  Address withPointer(llvm::Value *NewPointer) const {
    return Address(NewPointer, ElementType, Alignment);
  }

  /// Return address with different alignment, but same pointer and element
  /// type.
  Address withAlignment(CharUnits NewAlignment) const {
    return Address(Pointer, ElementType, NewAlignment);
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

  ConstantAddress getElementBitCast(llvm::Type *ElemTy) const {
    llvm::Constant *BitCast = llvm::ConstantExpr::getBitCast(
        getPointer(), ElemTy->getPointerTo(getAddressSpace()));
    return ConstantAddress(BitCast, ElemTy, getAlignment());
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

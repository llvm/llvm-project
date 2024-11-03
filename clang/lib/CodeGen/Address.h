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

// We try to save some space by using 6 bits over two PointerIntPairs to store
// the alignment. However, some arches don't support 3 bits in a PointerIntPair
// so we fallback to storing the alignment separately.
template <typename T, bool = alignof(llvm::Value *) >= 8> class AddressImpl {};

template <typename T> class AddressImpl<T, false> {
  llvm::PointerIntPair<llvm::Value *, 1, bool> PointerAndKnownNonNull;
  llvm::Type *ElementType;
  CharUnits Alignment;

public:
  AddressImpl(llvm::Value *Pointer, llvm::Type *ElementType,
              CharUnits Alignment, KnownNonNull_t IsKnownNonNull)
      : PointerAndKnownNonNull(Pointer, IsKnownNonNull),
        ElementType(ElementType), Alignment(Alignment) {}
  llvm::Value *getPointer() const {
    return PointerAndKnownNonNull.getPointer();
  }
  llvm::Type *getElementType() const { return ElementType; }
  CharUnits getAlignment() const { return Alignment; }
  KnownNonNull_t isKnownNonNull() const {
    return (KnownNonNull_t)PointerAndKnownNonNull.getInt();
  }
  void setKnownNonNull() { PointerAndKnownNonNull.setInt(true); }
};

template <typename T> class AddressImpl<T, true> {
  // Int portion stores the non-null bit and the upper 2 bits of the log of the
  // alignment.
  llvm::PointerIntPair<llvm::Value *, 3, unsigned> Pointer;
  // Int portion stores lower 3 bits of the log of the alignment.
  llvm::PointerIntPair<llvm::Type *, 3, unsigned> ElementType;

public:
  AddressImpl(llvm::Value *Pointer, llvm::Type *ElementType,
              CharUnits Alignment, KnownNonNull_t IsKnownNonNull)
      : Pointer(Pointer), ElementType(ElementType) {
    if (Alignment.isZero()) {
      this->Pointer.setInt(IsKnownNonNull << 2);
      return;
    }
    // Currently the max supported alignment is exactly 1 << 32 and is
    // guaranteed to be a power of 2, so we can store the log of the alignment
    // into 5 bits.
    assert(Alignment.isPowerOfTwo() && "Alignment cannot be zero");
    auto AlignLog = llvm::Log2_64(Alignment.getQuantity());
    assert(AlignLog < (1 << 5) && "cannot fit alignment into 5 bits");
    this->Pointer.setInt(IsKnownNonNull << 2 | AlignLog >> 3);
    this->ElementType.setInt(AlignLog & 7);
  }
  llvm::Value *getPointer() const { return Pointer.getPointer(); }
  llvm::Type *getElementType() const { return ElementType.getPointer(); }
  CharUnits getAlignment() const {
    unsigned AlignLog = ((Pointer.getInt() & 0x3) << 3) | ElementType.getInt();
    return CharUnits::fromQuantity(CharUnits::QuantityType(1) << AlignLog);
  }
  KnownNonNull_t isKnownNonNull() const {
    return (KnownNonNull_t)(!!(Pointer.getInt() & 0x4));
  }
  void setKnownNonNull() { Pointer.setInt(Pointer.getInt() | 0x4); }
};

/// An aligned address.
class Address {
  AddressImpl<void> A;

protected:
  Address(std::nullptr_t)
      : A(nullptr, nullptr, CharUnits::Zero(), NotKnownNonNull) {}

public:
  Address(llvm::Value *Pointer, llvm::Type *ElementType, CharUnits Alignment,
          KnownNonNull_t IsKnownNonNull = NotKnownNonNull)
      : A(Pointer, ElementType, Alignment, IsKnownNonNull) {
    assert(Pointer != nullptr && "Pointer cannot be null");
    assert(ElementType != nullptr && "Element type cannot be null");
    assert(llvm::cast<llvm::PointerType>(Pointer->getType())
               ->isOpaqueOrPointeeTypeMatches(ElementType) &&
           "Incorrect pointer element type");
  }

  static Address invalid() { return Address(nullptr); }
  bool isValid() const { return A.getPointer() != nullptr; }

  llvm::Value *getPointer() const {
    assert(isValid());
    return A.getPointer();
  }

  /// Return the type of the pointer value.
  llvm::PointerType *getType() const {
    return llvm::cast<llvm::PointerType>(getPointer()->getType());
  }

  /// Return the type of the values stored in this address.
  llvm::Type *getElementType() const {
    assert(isValid());
    return A.getElementType();
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
    return A.getAlignment();
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

  /// Whether the pointer is known not to be null.
  KnownNonNull_t isKnownNonNull() const {
    assert(isValid());
    return A.isKnownNonNull();
  }

  /// Set the non-null bit.
  Address setKnownNonNull() {
    assert(isValid());
    A.setKnownNonNull();
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

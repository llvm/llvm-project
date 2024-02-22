//===--- PointerAuthOptions.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines options for configuring pointer-auth technologies
//  like ARMv8.3.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_POINTERAUTHOPTIONS_H
#define LLVM_CLANG_BASIC_POINTERAUTHOPTIONS_H

#include "clang/Basic/LLVM.h"
#include "llvm/Target/TargetOptions.h"
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "llvm/Support/ErrorHandling.h"

namespace clang {

class PointerAuthSchema {
public:
  enum class Kind : unsigned {
    None,
    Soft,
    ARM8_3,
  };

  /// Software pointer-signing "keys". If you add a new key, make sure this->Key
  /// has a large enough bit-width.
  enum class SoftKey : unsigned {
    FunctionPointers = 0,
    BlockInvocationFunctionPointers = 1,
    BlockHelperFunctionPointers = 2,
    ObjCMethodListFunctionPointers = 3,
    CXXVTablePointers = 4,
    CXXVirtualFunctionPointers = 5,
    CXXMemberFunctionPointers = 6,
  };

  /// Hardware pointer-signing keys in ARM8.3.
  ///
  /// These values are the same used in ptrauth.h.
  enum class ARM8_3Key : unsigned {
    ASIA = 0,
    ASIB = 1,
    ASDA = 2,
    ASDB = 3
  };

  /// Forms of extra discrimination.
  enum class Discrimination : unsigned {
    /// No additional discrimination.
    None,

    /// Include a hash of the entity's type.
    Type,

    /// Include a hash of the entity's identity.
    Decl,

    /// Discriminate using a constant value.
    Constant,
  };

private:
  Kind TheKind : 2;
  unsigned IsAddressDiscriminated : 1;
  Discrimination DiscriminationKind : 2;
  unsigned Key : 3;
  unsigned ConstantDiscriminator : 16;

public:
  PointerAuthSchema() : TheKind(Kind::None) {}

  PointerAuthSchema(
      SoftKey key, bool isAddressDiscriminated,
      Discrimination otherDiscrimination,
      std::optional<uint16_t> constantDiscriminator = std::nullopt)
      : TheKind(Kind::Soft), IsAddressDiscriminated(isAddressDiscriminated),
        DiscriminationKind(otherDiscrimination), Key(unsigned(key)) {
    assert((getOtherDiscrimination() != Discrimination::Constant ||
            constantDiscriminator) &&
           "constant discrimination requires a constant!");
    if (constantDiscriminator)
      ConstantDiscriminator = *constantDiscriminator;
  }

  PointerAuthSchema(
      ARM8_3Key key, bool isAddressDiscriminated,
      Discrimination otherDiscrimination,
      std::optional<uint16_t> constantDiscriminator = std::nullopt)
      : TheKind(Kind::ARM8_3), IsAddressDiscriminated(isAddressDiscriminated),
        DiscriminationKind(otherDiscrimination), Key(unsigned(key)) {
    assert((getOtherDiscrimination() != Discrimination::Constant ||
            constantDiscriminator) &&
           "constant discrimination requires a constant!");
    if (constantDiscriminator)
      ConstantDiscriminator = *constantDiscriminator;
  }

  Kind getKind() const { return TheKind; }

  explicit operator bool() const { return isEnabled(); }

  bool isEnabled() const { return getKind() != Kind::None; }

  bool isAddressDiscriminated() const {
    assert(getKind() != Kind::None);
    return IsAddressDiscriminated;
  }

  bool hasOtherDiscrimination() const {
    return getOtherDiscrimination() != Discrimination::None;
  }

  Discrimination getOtherDiscrimination() const {
    assert(getKind() != Kind::None);
    return DiscriminationKind;
  }

  uint16_t getConstantDiscrimination() const {
    assert(getOtherDiscrimination() == Discrimination::Constant);
    return (uint16_t)ConstantDiscriminator;
  }

  unsigned getKey() const {
    switch (getKind()) {
    case Kind::None:
      llvm_unreachable("calling getKey() on disabled schema");
    case Kind::Soft:
      return unsigned(getSoftKey());
    case Kind::ARM8_3:
      return unsigned(getARM8_3Key());
    }
    llvm_unreachable("bad key kind");
  }

  SoftKey getSoftKey() const {
    assert(getKind() == Kind::Soft);
    return SoftKey(Key);
  }

  ARM8_3Key getARM8_3Key() const {
    assert(getKind() == Kind::ARM8_3);
    return ARM8_3Key(Key);
  }
};

struct PointerAuthOptions {
  /// Do member function pointers to virtual functions need to be built
  /// as thunks?
  bool ThunkCXXVirtualMemberPointers = false;

  /// Should return addresses be authenticated?
  bool ReturnAddresses = false;

  /// Do indirect goto label addresses need to be authenticated?
  bool IndirectGotos = false;

  /// Do authentication failures cause a trap?
  bool AuthTraps = false;

  /// The ABI for C function pointers.
  PointerAuthSchema FunctionPointers;

  /// The ABI for block invocation function pointers.
  PointerAuthSchema BlockInvocationFunctionPointers;

  /// The ABI for block object copy/destroy function pointers.
  PointerAuthSchema BlockHelperFunctionPointers;

  /// The ABI for __block variable copy/destroy function pointers.
  PointerAuthSchema BlockByrefHelperFunctionPointers;

  /// The ABI for Objective-C method lists.
  PointerAuthSchema ObjCMethodListFunctionPointers;

  /// The ABI for C++ virtual table pointers (the pointer to the table
  /// itself) as installed in an actual class instance.
  PointerAuthSchema CXXVTablePointers;

  /// The ABI for C++ virtual table pointers as installed in a VTT.
  PointerAuthSchema CXXVTTVTablePointers;

  /// The ABI for most C++ virtual function pointers, i.e. v-table entries.
  PointerAuthSchema CXXVirtualFunctionPointers;

  /// The ABI for variadic C++ virtual function pointers.
  PointerAuthSchema CXXVirtualVariadicFunctionPointers;

  /// The ABI for C++ member function pointers.
  PointerAuthSchema CXXMemberFunctionPointers;
};

}  // end namespace clang

#endif

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
#include "llvm/ADT/Optional.h"
#include "llvm/Target/TargetOptions.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "llvm/Support/ErrorHandling.h"

namespace clang {

class PointerAuthSchema {
public:
  enum class Kind {
    None,
    ARM8_3,
  };

  /// Hardware pointer-signing keys in ARM8.3.
  ///
  /// These values are the same used in ptrauth.h.
  enum class ARM8_3Key {
    ASIA = 0,
    ASIB = 1,
    ASDA = 2,
    ASDB = 3
  };

  /// Forms of extra discrimination.
  enum class Discrimination {
    /// No additional discrimination.
    None,

    /// Include a hash of the entity's type.
    Type,

    /// Include a hash of the entity's identity.
    Decl,
  };

private:
  enum {
    NumKindBits = 2
  };
  union {
    /// A common header shared by all pointer authentication kinds.
    struct {
      unsigned Kind : NumKindBits;
      unsigned AddressDiscriminated : 1;
      unsigned Discrimination : 2;
    } Common;

    struct {
      unsigned Kind : NumKindBits;
      unsigned AddressDiscriminated : 1;
      unsigned Discrimination : 2;
      unsigned Key : 2;
    } ARM8_3;
  };

public:
  PointerAuthSchema() {
    Common.Kind = unsigned(Kind::None);
  }

  PointerAuthSchema(ARM8_3Key key, bool isAddressDiscriminated,
                    Discrimination otherDiscrimination) {
    Common.Kind = unsigned(Kind::ARM8_3);
    Common.AddressDiscriminated = isAddressDiscriminated;
    Common.Discrimination = unsigned(otherDiscrimination);
    ARM8_3.Key = unsigned(key);
  }

  Kind getKind() const {
    return Kind(Common.Kind);
  }

  explicit operator bool() const {
    return isEnabled();
  }

  bool isEnabled() const {
    return getKind() != Kind::None;
  }

  bool isAddressDiscriminated() const {
    assert(getKind() != Kind::None);
    return Common.AddressDiscriminated;
  }

  bool hasOtherDiscrimination() const {
    return getOtherDiscrimination() != Discrimination::None;
  }

  Discrimination getOtherDiscrimination() const {
    assert(getKind() != Kind::None);
    return Discrimination(Common.Discrimination);
  }

  unsigned getKey() const {
    switch (getKind()) {
    case Kind::None: llvm_unreachable("calling getKey() on disabled schema");
    case Kind::ARM8_3: return unsigned(getARM8_3Key());
    }
    llvm_unreachable("bad key kind");
  }

  ARM8_3Key getARM8_3Key() const {
    assert(getKind() == Kind::ARM8_3);
    return ARM8_3Key(ARM8_3.Key);
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
};

}  // end namespace clang

#endif

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
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetOptions.h"
#include <optional>

namespace clang {

/// Constant discriminator to be used with block descriptor pointers. The value
/// is ptrauth_string_discriminator("block_descriptor")
constexpr uint16_t BlockDescriptorConstantDiscriminator = 0xC0BB;

/// Constant discriminator to be used with function pointers in .init_array and
/// .fini_array. The value is ptrauth_string_discriminator("init_fini")
constexpr uint16_t InitFiniPointerConstantDiscriminator = 0xD9D4;

/// Constant discriminator to be used with method list pointers. The value is
/// ptrauth_string_discriminator("method_list_t")
constexpr uint16_t MethodListPointerConstantDiscriminator = 0xC310;

/// Constant discriminator to be used with objective-c isa pointers. The value
/// is ptrauth_string_discriminator("isa")
constexpr uint16_t IsaPointerConstantDiscriminator = 0x6AE1;

/// Constant discriminator to be used with objective-c superclass pointers.
/// The value is ptrauth_string_discriminator("objc_class:superclass")
constexpr uint16_t SuperPointerConstantDiscriminator = 0xB5AB;

/// Constant discriminator to be used with objective-c sel pointers. The value
/// is ptrauth_string_discriminator("sel")
constexpr uint16_t SelPointerConstantDiscriminator = 0x57c2;

/// Constant discriminator to be used with objective-c class_ro_t pointers.
/// The value is ptrauth_string_discriminator("class_data_bits")
constexpr uint16_t ClassROConstantDiscriminator = 0x61F8;

constexpr unsigned PointerAuthKeyNone = -1;

/// Constant discriminator for std::type_info vtable pointers: 0xB1EA/45546
/// The value is ptrauth_string_discriminator("_ZTVSt9type_info"), i.e.,
/// the vtable type discriminator for classes derived from std::type_info.
constexpr uint16_t StdTypeInfoVTablePointerConstantDiscrimination = 0xB1EA;

class PointerAuthSchema {
public:
  enum class Kind : unsigned {
    None,
    ARM8_3,
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
  unsigned IsIsaPointer : 1;
  unsigned AuthenticatesNullValues : 1;
  PointerAuthenticationMode SelectedAuthenticationMode : 2;
  Discrimination DiscriminationKind : 2;
  unsigned Key : 2;
  unsigned ConstantDiscriminator : 16;

public:
  PointerAuthSchema() : TheKind(Kind::None) {}

  PointerAuthSchema(
      ARM8_3Key Key, bool IsAddressDiscriminated,
      PointerAuthenticationMode AuthenticationMode,
      Discrimination OtherDiscrimination,
      std::optional<uint16_t> ConstantDiscriminatorOrNone = std::nullopt,
      bool IsIsaPointer = false, bool AuthenticatesNullValues = false)
      : TheKind(Kind::ARM8_3), IsAddressDiscriminated(IsAddressDiscriminated),
        IsIsaPointer(IsIsaPointer),
        AuthenticatesNullValues(AuthenticatesNullValues),
        SelectedAuthenticationMode(AuthenticationMode),
        DiscriminationKind(OtherDiscrimination), Key(llvm::to_underlying(Key)) {
    assert((getOtherDiscrimination() != Discrimination::Constant ||
            ConstantDiscriminatorOrNone) &&
           "constant discrimination requires a constant!");
    if (ConstantDiscriminatorOrNone)
      ConstantDiscriminator = *ConstantDiscriminatorOrNone;
  }

  PointerAuthSchema(
      ARM8_3Key Key, bool IsAddressDiscriminated,
      Discrimination OtherDiscrimination,
      std::optional<uint16_t> ConstantDiscriminatorOrNone = std::nullopt,
      bool IsIsaPointer = false, bool AuthenticatesNullValues = false)
      : PointerAuthSchema(Key, IsAddressDiscriminated,
                          PointerAuthenticationMode::SignAndAuth,
                          OtherDiscrimination, ConstantDiscriminatorOrNone,
                          IsIsaPointer, AuthenticatesNullValues) {}

  Kind getKind() const { return TheKind; }

  explicit operator bool() const { return isEnabled(); }

  bool isEnabled() const { return getKind() != Kind::None; }

  bool isAddressDiscriminated() const {
    assert(getKind() != Kind::None);
    return IsAddressDiscriminated;
  }

  bool isIsaPointer() const {
    assert(getKind() != Kind::None);
    return IsIsaPointer;
  }

  bool authenticatesNullValues() const {
    assert(getKind() != Kind::None);
    return AuthenticatesNullValues;
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
    return ConstantDiscriminator;
  }

  unsigned getKey() const {
    switch (getKind()) {
    case Kind::None:
      llvm_unreachable("calling getKey() on disabled schema");
    case Kind::ARM8_3:
      return llvm::to_underlying(getARM8_3Key());
    }
    llvm_unreachable("bad key kind");
  }

  PointerAuthenticationMode getAuthenticationMode() const {
    return SelectedAuthenticationMode;
  }

  ARM8_3Key getARM8_3Key() const {
    assert(getKind() == Kind::ARM8_3);
    return ARM8_3Key(Key);
  }
};

struct PointerAuthOptions {
  /// Should return addresses be authenticated?
  bool ReturnAddresses = false;

  /// Do authentication failures cause a trap?
  bool AuthTraps = false;

  /// Do indirect goto label addresses need to be authenticated?
  bool IndirectGotos = false;

  /// Use hardened lowering for jump-table dispatch?
  bool AArch64JumpTableHardening = false;

  /// The ABI for C function pointers.
  PointerAuthSchema FunctionPointers;

  /// The ABI for C++ virtual table pointers (the pointer to the table
  /// itself) as installed in an actual class instance.
  PointerAuthSchema CXXVTablePointers;

  /// TypeInfo has external ABI requirements and is emitted without
  /// actually having parsed the libcxx definition, so we can't simply
  /// perform a look up. The settings for this should match the exact
  /// specification in type_info.h
  PointerAuthSchema CXXTypeInfoVTablePointer;

  /// The ABI for C++ virtual table pointers as installed in a VTT.
  PointerAuthSchema CXXVTTVTablePointers;

  /// The ABI for most C++ virtual function pointers, i.e. v-table entries.
  PointerAuthSchema CXXVirtualFunctionPointers;

  /// The ABI for variadic C++ virtual function pointers.
  PointerAuthSchema CXXVirtualVariadicFunctionPointers;

  /// The ABI for C++ member function pointers.
  PointerAuthSchema CXXMemberFunctionPointers;

  /// The ABI for function addresses in .init_array and .fini_array
  PointerAuthSchema InitFiniPointers;

  /// The ABI for block invocation function pointers.
  PointerAuthSchema BlockInvocationFunctionPointers;

  /// The ABI for block object copy/destroy function pointers.
  PointerAuthSchema BlockHelperFunctionPointers;

  /// The ABI for __block variable copy/destroy function pointers.
  PointerAuthSchema BlockByrefHelperFunctionPointers;

  /// The ABI for pointers to block descriptors.
  PointerAuthSchema BlockDescriptorPointers;

  /// The ABI for Objective-C method lists.
  PointerAuthSchema ObjCMethodListFunctionPointers;

  /// The ABI for a reference to an Objective-C method list in _class_ro_t.
  PointerAuthSchema ObjCMethodListPointer;

  /// The ABI for Objective-C isa pointers.
  PointerAuthSchema ObjCIsaPointers;

  /// The ABI for Objective-C superclass pointers.
  PointerAuthSchema ObjCSuperPointers;

  /// The ABI for Objective-C class_ro_t pointers.
  PointerAuthSchema ObjCClassROPointers;
};

} // end namespace clang

#endif

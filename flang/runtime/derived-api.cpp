//===-- runtime/derived-api.cpp
//-----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/derived-api.h"
#include "derived.h"
#include "terminator.h"
#include "tools.h"
#include "type-info.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime {

extern "C" {

void RTDEF(Initialize)(
    const Descriptor &descriptor, const char *sourceFile, int sourceLine) {
  if (const DescriptorAddendum * addendum{descriptor.Addendum()}) {
    if (const auto *derived{addendum->derivedType()}) {
      if (!derived->noInitializationNeeded()) {
        Terminator terminator{sourceFile, sourceLine};
        Initialize(descriptor, *derived, terminator);
      }
    }
  }
}

void RTDEF(Destroy)(const Descriptor &descriptor) {
  if (const DescriptorAddendum * addendum{descriptor.Addendum()}) {
    if (const auto *derived{addendum->derivedType()}) {
      if (!derived->noDestructionNeeded()) {
        // TODO: Pass source file & line information to the API
        // so that a good Terminator can be passed
        Destroy(descriptor, true, *derived, nullptr);
      }
    }
  }
}

void RTDEF(Finalize)(
    const Descriptor &descriptor, const char *sourceFile, int sourceLine) {
  if (const DescriptorAddendum * addendum{descriptor.Addendum()}) {
    if (const auto *derived{addendum->derivedType()}) {
      if (!derived->noFinalizationNeeded()) {
        Terminator terminator{sourceFile, sourceLine};
        Finalize(descriptor, *derived, &terminator);
      }
    }
  }
}

bool RTDEF(ClassIs)(
    const Descriptor &descriptor, const typeInfo::DerivedType &derivedType) {
  if (const DescriptorAddendum * addendum{descriptor.Addendum()}) {
    if (const auto *derived{addendum->derivedType()}) {
      if (derived == &derivedType) {
        return true;
      }
      const typeInfo::DerivedType *parent{derived->GetParentType()};
      while (parent) {
        if (parent == &derivedType) {
          return true;
        }
        parent = parent->GetParentType();
      }
    }
  }
  return false;
}

static RT_API_ATTRS bool CompareDerivedTypeNames(
    const Descriptor &a, const Descriptor &b) {
  if (a.raw().version == CFI_VERSION &&
      a.type() == TypeCode{TypeCategory::Character, 1} &&
      a.ElementBytes() > 0 && a.rank() == 0 && a.OffsetElement() != nullptr &&
      a.raw().version == CFI_VERSION &&
      b.type() == TypeCode{TypeCategory::Character, 1} &&
      b.ElementBytes() > 0 && b.rank() == 0 && b.OffsetElement() != nullptr &&
      a.ElementBytes() == b.ElementBytes() &&
      Fortran::runtime::memcmp(
          a.OffsetElement(), b.OffsetElement(), a.ElementBytes()) == 0) {
    return true;
  }
  return false;
}

inline RT_API_ATTRS bool CompareDerivedType(
    const typeInfo::DerivedType *a, const typeInfo::DerivedType *b) {
  return a == b || CompareDerivedTypeNames(a->name(), b->name());
}

static const RT_API_ATTRS typeInfo::DerivedType *GetDerivedType(
    const Descriptor &desc) {
  if (const DescriptorAddendum * addendum{desc.Addendum()}) {
    if (const auto *derived{addendum->derivedType()}) {
      return derived;
    }
  }
  return nullptr;
}

bool RTDEF(SameTypeAs)(const Descriptor &a, const Descriptor &b) {
  auto aType{a.raw().type};
  auto bType{b.raw().type};
  if ((aType != CFI_type_struct && aType != CFI_type_other) ||
      (bType != CFI_type_struct && bType != CFI_type_other)) {
    // If either type is intrinsic, they must match.
    return aType == bType;
  } else {
    const typeInfo::DerivedType *derivedTypeA{GetDerivedType(a)};
    const typeInfo::DerivedType *derivedTypeB{GetDerivedType(b)};
    if (derivedTypeA == nullptr || derivedTypeB == nullptr) {
      // Unallocated/disassociated CLASS(*) never matches.
      return false;
    } else if (derivedTypeA == derivedTypeB) {
      // Exact match of derived type.
      return true;
    } else {
      // Otherwise compare with the name. Note 16.29 kind type parameters are
      // not considered in the test.
      return CompareDerivedTypeNames(
          derivedTypeA->name(), derivedTypeB->name());
    }
  }
}

bool RTDEF(ExtendsTypeOf)(const Descriptor &a, const Descriptor &mold) {
  auto aType{a.raw().type};
  auto moldType{mold.raw().type};
  if ((aType != CFI_type_struct && aType != CFI_type_other) ||
      (moldType != CFI_type_struct && moldType != CFI_type_other)) {
    // If either type is intrinsic, they must match.
    return aType == moldType;
  } else if (const typeInfo::DerivedType *
      derivedTypeMold{GetDerivedType(mold)}) {
    // If A is unlimited polymorphic and is either a disassociated pointer or
    // unallocated allocatable, the result is false.
    // Otherwise if the dynamic type of A or MOLD is extensible, the result is
    // true if and only if the dynamic type of A is an extension type of the
    // dynamic type of MOLD.
    for (const typeInfo::DerivedType *derivedTypeA{GetDerivedType(a)};
         derivedTypeA; derivedTypeA = derivedTypeA->GetParentType()) {
      if (CompareDerivedType(derivedTypeA, derivedTypeMold)) {
        return true;
      }
    }
    return false;
  } else {
    // MOLD is unlimited polymorphic and unallocated/disassociated.
    return true;
  }
}

void RTDEF(DestroyWithoutFinalization)(const Descriptor &descriptor) {
  if (const DescriptorAddendum * addendum{descriptor.Addendum()}) {
    if (const auto *derived{addendum->derivedType()}) {
      if (!derived->noDestructionNeeded()) {
        Destroy(descriptor, /*finalize=*/false, *derived, nullptr);
      }
    }
  }
}

} // extern "C"
} // namespace Fortran::runtime

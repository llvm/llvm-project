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
#include "type-info.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime {

extern "C" {

void RTNAME(Initialize)(
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

void RTNAME(Destroy)(const Descriptor &descriptor) {
  if (const DescriptorAddendum * addendum{descriptor.Addendum()}) {
    if (const auto *derived{addendum->derivedType()}) {
      if (!derived->noDestructionNeeded()) {
        Destroy(descriptor, true, *derived);
      }
    }
  }
}

bool RTNAME(ClassIs)(
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

static bool CompareDerivedTypeNames(const Descriptor &a, const Descriptor &b) {
  if (a.raw().version == CFI_VERSION &&
      a.type() == TypeCode{TypeCategory::Character, 1} &&
      a.ElementBytes() > 0 && a.rank() == 0 && a.OffsetElement() != nullptr &&
      a.raw().version == CFI_VERSION &&
      b.type() == TypeCode{TypeCategory::Character, 1} &&
      b.ElementBytes() > 0 && b.rank() == 0 && b.OffsetElement() != nullptr &&
      a.ElementBytes() == b.ElementBytes() &&
      memcmp(a.OffsetElement(), b.OffsetElement(), a.ElementBytes()) == 0) {
    return true;
  }
  return false;
}

inline bool CompareDerivedType(
    const typeInfo::DerivedType *a, const typeInfo::DerivedType *b) {
  return a == b || CompareDerivedTypeNames(a->name(), b->name());
}

static const typeInfo::DerivedType *GetDerivedType(const Descriptor &desc) {
  if (const DescriptorAddendum * addendum{desc.Addendum()}) {
    if (const auto *derived{addendum->derivedType()}) {
      return derived;
    }
  }
  return nullptr;
}

bool RTNAME(SameTypeAs)(const Descriptor &a, const Descriptor &b) {
  const typeInfo::DerivedType *derivedTypeA{GetDerivedType(a)};
  const typeInfo::DerivedType *derivedTypeB{GetDerivedType(b)};
  if (derivedTypeA == nullptr || derivedTypeB == nullptr) {
    return false;
  }
  // Exact match of derived type.
  if (derivedTypeA == derivedTypeB) {
    return true;
  }
  // Otherwise compare with the name. Note 16.29 kind type parameters are not
  // considered in the test.
  return CompareDerivedTypeNames(derivedTypeA->name(), derivedTypeB->name());
}

bool RTNAME(ExtendsTypeOf)(const Descriptor &a, const Descriptor &mold) {
  const typeInfo::DerivedType *derivedTypeA{GetDerivedType(a)};
  const typeInfo::DerivedType *derivedTypeMold{GetDerivedType(mold)};

  // If MOLD is unlimited polymorphic and is either a disassociated pointer or
  // unallocated allocatable, the result is true.
  // Unlimited polymorphic descriptors are initialized with a CFI_type_other
  // type.
  if (mold.type().raw() == CFI_type_other &&
      (mold.IsAllocatable() || mold.IsPointer()) &&
      derivedTypeMold == nullptr) {
    return true;
  }

  // If A is unlimited polymorphic and is either a disassociated pointer or
  // unallocated allocatable, the result is false.
  // Unlimited polymorphic descriptors are initialized with a CFI_type_other
  // type.
  if (a.type().raw() == CFI_type_other &&
      (a.IsAllocatable() || a.IsPointer()) && derivedTypeA == nullptr) {
    return false;
  }

  if (derivedTypeA == nullptr || derivedTypeMold == nullptr) {
    return false;
  }

  // Otherwise if the dynamic type of A or MOLD is extensible, the result is
  // true if and only if the dynamic type of A is an extension type of the
  // dynamic type of MOLD.
  if (CompareDerivedType(derivedTypeA, derivedTypeMold)) {
    return true;
  }
  const typeInfo::DerivedType *parent{derivedTypeA->GetParentType()};
  while (parent) {
    if (CompareDerivedType(parent, derivedTypeMold)) {
      return true;
    }
    parent = parent->GetParentType();
  }
  return false;
}

// TODO: Assign()

} // extern "C"
} // namespace Fortran::runtime

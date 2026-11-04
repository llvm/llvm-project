//===-- lib/runtime/derived-api.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/derived-api.h"
#include "flang-rt/runtime/derived.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/tools.h"
#include "flang-rt/runtime/type-info.h"

namespace Fortran::runtime {

extern "C" {
RT_EXT_API_GROUP_BEGIN

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

void RTDEF(InitializeClone)(const Descriptor &clone, const Descriptor &orig,
    const char *sourceFile, int sourceLine) {
  if (const DescriptorAddendum * addendum{clone.Addendum()}) {
    if (const auto *derived{addendum->derivedType()}) {
      Terminator terminator{sourceFile, sourceLine};
      InitializeClone(clone, orig, *derived, terminator);
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

static RT_API_ATTRS const typeInfo::DerivedType *GetDerivedType(
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
  } else if (const typeInfo::DerivedType * derivedTypeA{GetDerivedType(a)}) {
    if (const typeInfo::DerivedType * derivedTypeB{GetDerivedType(b)}) {
      if (derivedTypeA == derivedTypeB) {
        return true;
      } else if (const typeInfo::DerivedType *
          uninstDerivedTypeA{derivedTypeA->uninstantiatedType()}) {
        // There are KIND type parameters, are these the same type if those
        // are ignored?
        const typeInfo::DerivedType *uninstDerivedTypeB{
            derivedTypeB->uninstantiatedType()};
        return uninstDerivedTypeA == uninstDerivedTypeB;
      }
    }
  }
  return false;
}

bool RTDEF(ExtendsTypeOf)(const Descriptor &a, const Descriptor &mold) {
  // The wording of the standard indicates null or unallocated checks take
  // precedence over the extension checks which take precedence over any
  // compiler specific behavior.
  // F'23 16.9.86 p 5
  // If MOLD is unlimited polymorphic and is either a disassociated pointer or
  // unallocated allocatable variable, the result is true;
  auto aType{a.raw().type};
  auto moldType{mold.raw().type};
  if ((aType != CFI_type_struct && aType != CFI_type_other) ||
      (moldType != CFI_type_struct && moldType != CFI_type_other)) {
    if (!mold.IsAllocated()) {
      return true;
    } else if (!a.IsAllocated()) {
      return false;
    } else {
      // If either type is intrinsic and not a pointer or allocatable
      // then they must match.
      return aType == moldType;
    }
  } else if (const auto *derivedTypeMold{GetDerivedType(mold)}) {
    // If A is unlimited polymorphic and is either a disassociated pointer or
    // unallocated allocatable, the result is false.
    // Otherwise if the dynamic type of A or MOLD is extensible, the result is
    // true if and only if the dynamic type of A is an extension type of the
    // dynamic type of MOLD.
    for (const typeInfo::DerivedType *derivedTypeA{GetDerivedType(a)};
         derivedTypeA; derivedTypeA = derivedTypeA->GetParentType()) {
      if (derivedTypeA == derivedTypeMold) {
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

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime

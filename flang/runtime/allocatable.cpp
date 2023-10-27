//===-- runtime/allocatable.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/allocatable.h"
#include "assign-impl.h"
#include "derived.h"
#include "stat.h"
#include "terminator.h"
#include "type-info.h"
#include "flang/ISO_Fortran_binding_wrapper.h"
#include "flang/Runtime/assign.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime {
extern "C" {

void RTNAME(AllocatableInitIntrinsic)(Descriptor &descriptor,
    TypeCategory category, int kind, int rank, int corank) {
  INTERNAL_CHECK(corank == 0);
  descriptor.Establish(TypeCode{category, kind},
      Descriptor::BytesFor(category, kind), nullptr, rank, nullptr,
      CFI_attribute_allocatable);
}

void RTNAME(AllocatableInitCharacter)(Descriptor &descriptor,
    SubscriptValue length, int kind, int rank, int corank) {
  INTERNAL_CHECK(corank == 0);
  descriptor.Establish(
      kind, length, nullptr, rank, nullptr, CFI_attribute_allocatable);
}

void RTNAME(AllocatableInitDerived)(Descriptor &descriptor,
    const typeInfo::DerivedType &derivedType, int rank, int corank) {
  INTERNAL_CHECK(corank == 0);
  descriptor.Establish(
      derivedType, nullptr, rank, nullptr, CFI_attribute_allocatable);
}

void RTNAME(AllocatableInitIntrinsicForAllocate)(Descriptor &descriptor,
    TypeCategory category, int kind, int rank, int corank) {
  if (descriptor.IsAllocated()) {
    return;
  }
  RTNAME(AllocatableInitIntrinsic)(descriptor, category, kind, rank, corank);
}

void RTNAME(AllocatableInitCharacterForAllocate)(Descriptor &descriptor,
    SubscriptValue length, int kind, int rank, int corank) {
  if (descriptor.IsAllocated()) {
    return;
  }
  RTNAME(AllocatableInitCharacter)(descriptor, length, kind, rank, corank);
}

void RTNAME(AllocatableInitDerivedForAllocate)(Descriptor &descriptor,
    const typeInfo::DerivedType &derivedType, int rank, int corank) {
  if (descriptor.IsAllocated()) {
    return;
  }
  RTNAME(AllocatableInitDerived)(descriptor, derivedType, rank, corank);
}

std::int32_t RTNAME(MoveAlloc)(Descriptor &to, Descriptor &from,
    const typeInfo::DerivedType *derivedType, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};

  // If to and from are the same allocatable they must not be allocated
  // and nothing should be done.
  if (from.raw().base_addr == to.raw().base_addr && from.IsAllocated()) {
    return ReturnError(
        terminator, StatMoveAllocSameAllocatable, errMsg, hasStat);
  }

  if (to.IsAllocated()) {
    int stat{
        to.Destroy(/*finalize=*/true, /*destroyPointers=*/false, &terminator)};
    if (stat != StatOk) {
      return ReturnError(terminator, stat, errMsg, hasStat);
    }
  }

  // If from isn't allocated, the standard defines that nothing should be done.
  if (from.IsAllocated()) {
    to = from;
    from.raw().base_addr = nullptr;

    // Carry over the dynamic type.
    if (auto *toAddendum{to.Addendum()}) {
      if (const auto *fromAddendum{from.Addendum()}) {
        if (const auto *derived{fromAddendum->derivedType()}) {
          toAddendum->set_derivedType(derived);
        }
      }
    }

    // Reset from dynamic type if needed.
    if (auto *fromAddendum{from.Addendum()}) {
      if (derivedType) {
        fromAddendum->set_derivedType(derivedType);
      }
    }
  }

  return StatOk;
}

void RTNAME(AllocatableSetBounds)(Descriptor &descriptor, int zeroBasedDim,
    SubscriptValue lower, SubscriptValue upper) {
  INTERNAL_CHECK(zeroBasedDim >= 0 && zeroBasedDim < descriptor.rank());
  descriptor.GetDimension(zeroBasedDim).SetBounds(lower, upper);
  // The byte strides are computed when the object is allocated.
}

void RTNAME(AllocatableSetDerivedLength)(
    Descriptor &descriptor, int which, SubscriptValue x) {
  DescriptorAddendum *addendum{descriptor.Addendum()};
  INTERNAL_CHECK(addendum != nullptr);
  addendum->SetLenParameterValue(which, x);
}

void RTNAME(AllocatableApplyMold)(
    Descriptor &descriptor, const Descriptor &mold, int rank) {
  if (descriptor.IsAllocated()) {
    // 9.7.1.3 Return so the error can be emitted by AllocatableAllocate.
    return;
  }
  descriptor = mold;
  descriptor.set_base_addr(nullptr);
  descriptor.raw().attribute = CFI_attribute_allocatable;
  descriptor.raw().rank = rank;
  if (auto *descAddendum{descriptor.Addendum()}) {
    if (const auto *moldAddendum{mold.Addendum()}) {
      if (const auto *derived{moldAddendum->derivedType()}) {
        descAddendum->set_derivedType(derived);
      }
    }
  }
}

int RTNAME(AllocatableAllocate)(Descriptor &descriptor, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (!descriptor.IsAllocatable()) {
    return ReturnError(terminator, StatInvalidDescriptor, errMsg, hasStat);
  }
  if (descriptor.IsAllocated()) {
    return ReturnError(terminator, StatBaseNotNull, errMsg, hasStat);
  }
  int stat{ReturnError(terminator, descriptor.Allocate(), errMsg, hasStat)};
  if (stat == StatOk) {
    if (const DescriptorAddendum * addendum{descriptor.Addendum()}) {
      if (const auto *derived{addendum->derivedType()}) {
        if (!derived->noInitializationNeeded()) {
          stat = Initialize(descriptor, *derived, terminator, hasStat, errMsg);
        }
      }
    }
  }
  return stat;
}

int RTNAME(AllocatableAllocateSource)(Descriptor &alloc,
    const Descriptor &source, bool hasStat, const Descriptor *errMsg,
    const char *sourceFile, int sourceLine) {
  int stat{RTNAME(AllocatableAllocate)(
      alloc, hasStat, errMsg, sourceFile, sourceLine)};
  if (stat == StatOk) {
    Terminator terminator{sourceFile, sourceLine};
    DoFromSourceAssign(alloc, source, terminator);
  }
  return stat;
}

int RTNAME(AllocatableDeallocate)(Descriptor &descriptor, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (!descriptor.IsAllocatable()) {
    return ReturnError(terminator, StatInvalidDescriptor, errMsg, hasStat);
  }
  if (!descriptor.IsAllocated()) {
    return ReturnError(terminator, StatBaseNull, errMsg, hasStat);
  }
  return ReturnError(terminator,
      descriptor.Destroy(
          /*finalize=*/true, /*destroyPointers=*/false, &terminator),
      errMsg, hasStat);
}

int RTNAME(AllocatableDeallocatePolymorphic)(Descriptor &descriptor,
    const typeInfo::DerivedType *derivedType, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  int stat{RTNAME(AllocatableDeallocate)(
      descriptor, hasStat, errMsg, sourceFile, sourceLine)};
  if (stat == StatOk) {
    DescriptorAddendum *addendum{descriptor.Addendum()};
    if (addendum) {
      addendum->set_derivedType(derivedType);
    } else {
      // Unlimited polymorphic descriptors initialized with
      // AllocatableInitIntrinsic do not have an addendum. Make sure the
      // derivedType is null in that case.
      INTERNAL_CHECK(!derivedType);
    }
  }
  return stat;
}

void RTNAME(AllocatableDeallocateNoFinal)(
    Descriptor &descriptor, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (!descriptor.IsAllocatable()) {
    ReturnError(terminator, StatInvalidDescriptor);
  } else if (!descriptor.IsAllocated()) {
    ReturnError(terminator, StatBaseNull);
  } else {
    ReturnError(terminator,
        descriptor.Destroy(
            /*finalize=*/false, /*destroyPointers=*/false, &terminator));
  }
}

// TODO: AllocatableCheckLengthParameter
}
} // namespace Fortran::runtime

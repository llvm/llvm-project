//===-- runtime/allocatable.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "allocatable.h"
#include "stat.h"
#include "terminator.h"

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

void RTNAME(AllocatableAssign)(Descriptor &to, const Descriptor & /*from*/) {
  INTERNAL_CHECK(false); // AllocatableAssign is not yet implemented
}

int RTNAME(MoveAlloc)(Descriptor &to, const Descriptor & /*from*/,
    bool /*hasStat*/, const Descriptor * /*errMsg*/,
    const char * /*sourceFile*/, int /*sourceLine*/) {
  INTERNAL_CHECK(false); // MoveAlloc is not yet implemented
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
    Descriptor &descriptor, const Descriptor &mold) {
  descriptor = mold;
  descriptor.set_base_addr(nullptr);
  descriptor.raw().attribute = CFI_attribute_allocatable;
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
  return ReturnError(terminator, descriptor.Allocate(), errMsg, hasStat);
  // TODO: default component initialization
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
  return ReturnError(terminator, descriptor.Deallocate(), errMsg, hasStat);
}

// TODO: AllocatableCheckLengthParameter, AllocatableAllocateSource
}
} // namespace Fortran::runtime

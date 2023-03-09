//===-- runtime/pointer.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/pointer.h"
#include "assign-impl.h"
#include "derived.h"
#include "stat.h"
#include "terminator.h"
#include "tools.h"
#include "type-info.h"

namespace Fortran::runtime {
extern "C" {

void RTNAME(PointerNullifyIntrinsic)(Descriptor &pointer, TypeCategory category,
    int kind, int rank, int corank) {
  INTERNAL_CHECK(corank == 0);
  pointer.Establish(TypeCode{category, kind},
      Descriptor::BytesFor(category, kind), nullptr, rank, nullptr,
      CFI_attribute_pointer);
}

void RTNAME(PointerNullifyCharacter)(Descriptor &pointer, SubscriptValue length,
    int kind, int rank, int corank) {
  INTERNAL_CHECK(corank == 0);
  pointer.Establish(
      kind, length, nullptr, rank, nullptr, CFI_attribute_pointer);
}

void RTNAME(PointerNullifyDerived)(Descriptor &pointer,
    const typeInfo::DerivedType &derivedType, int rank, int corank) {
  INTERNAL_CHECK(corank == 0);
  pointer.Establish(derivedType, nullptr, rank, nullptr, CFI_attribute_pointer);
}

void RTNAME(PointerSetBounds)(Descriptor &pointer, int zeroBasedDim,
    SubscriptValue lower, SubscriptValue upper) {
  INTERNAL_CHECK(zeroBasedDim >= 0 && zeroBasedDim < pointer.rank());
  pointer.GetDimension(zeroBasedDim).SetBounds(lower, upper);
  // The byte strides are computed when the pointer is allocated.
}

// TODO: PointerSetCoBounds

void RTNAME(PointerSetDerivedLength)(
    Descriptor &pointer, int which, SubscriptValue x) {
  DescriptorAddendum *addendum{pointer.Addendum()};
  INTERNAL_CHECK(addendum != nullptr);
  addendum->SetLenParameterValue(which, x);
}

void RTNAME(PointerApplyMold)(
    Descriptor &pointer, const Descriptor &mold, int rank) {
  pointer = mold;
  pointer.set_base_addr(nullptr);
  pointer.raw().attribute = CFI_attribute_pointer;
  pointer.raw().rank = rank;
  if (auto *pointerAddendum{pointer.Addendum()}) {
    if (const auto *moldAddendum{mold.Addendum()}) {
      if (const auto *derived{moldAddendum->derivedType()}) {
        pointerAddendum->set_derivedType(derived);
      }
    }
  }
}

void RTNAME(PointerAssociateScalar)(Descriptor &pointer, void *target) {
  pointer.set_base_addr(target);
}

void RTNAME(PointerAssociate)(Descriptor &pointer, const Descriptor &target) {
  pointer = target;
  pointer.raw().attribute = CFI_attribute_pointer;
}

void RTNAME(PointerAssociateLowerBounds)(Descriptor &pointer,
    const Descriptor &target, const Descriptor &lowerBounds) {
  pointer = target;
  pointer.raw().attribute = CFI_attribute_pointer;
  int rank{pointer.rank()};
  Terminator terminator{__FILE__, __LINE__};
  std::size_t boundElementBytes{lowerBounds.ElementBytes()};
  for (int j{0}; j < rank; ++j) {
    Dimension &dim{pointer.GetDimension(j)};
    dim.SetLowerBound(dim.Extent() == 0
            ? 1
            : GetInt64(lowerBounds.ZeroBasedIndexedElement<const char>(j),
                  boundElementBytes, terminator));
  }
}

void RTNAME(PointerAssociateRemapping)(Descriptor &pointer,
    const Descriptor &target, const Descriptor &bounds, const char *sourceFile,
    int sourceLine) {
  pointer = target;
  pointer.raw().attribute = CFI_attribute_pointer;
  Terminator terminator{sourceFile, sourceLine};
  SubscriptValue byteStride{/*captured from first dimension*/};
  std::size_t boundElementBytes{bounds.ElementBytes()};
  std::size_t boundsRank{
      static_cast<std::size_t>(bounds.GetDimension(1).Extent())};
  pointer.raw().rank = boundsRank;
  for (unsigned j{0}; j < boundsRank; ++j) {
    auto &dim{pointer.GetDimension(j)};
    dim.SetBounds(GetInt64(bounds.ZeroBasedIndexedElement<const char>(2 * j),
                      boundElementBytes, terminator),
        GetInt64(bounds.ZeroBasedIndexedElement<const char>(2 * j + 1),
            boundElementBytes, terminator));
    if (j == 0) {
      byteStride = dim.ByteStride() * dim.Extent();
    } else {
      dim.SetByteStride(byteStride);
      byteStride *= dim.Extent();
    }
  }
  if (pointer.Elements() > target.Elements()) {
    terminator.Crash("PointerAssociateRemapping: too many elements in remapped "
                     "pointer (%zd > %zd)",
        pointer.Elements(), target.Elements());
  }
  if (auto *pointerAddendum{pointer.Addendum()}) {
    if (const auto *targetAddendum{target.Addendum()}) {
      if (const auto *derived{targetAddendum->derivedType()}) {
        pointerAddendum->set_derivedType(derived);
      }
    }
  }
}

int RTNAME(PointerAllocate)(Descriptor &pointer, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (!pointer.IsPointer()) {
    return ReturnError(terminator, StatInvalidDescriptor, errMsg, hasStat);
  }
  int stat{ReturnError(terminator, pointer.Allocate(), errMsg, hasStat)};
  if (stat == StatOk) {
    if (const DescriptorAddendum * addendum{pointer.Addendum()}) {
      if (const auto *derived{addendum->derivedType()}) {
        if (!derived->noInitializationNeeded()) {
          stat = Initialize(pointer, *derived, terminator, hasStat, errMsg);
        }
      }
    }
  }
  return stat;
}

int RTNAME(PointerAllocateSource)(Descriptor &pointer, const Descriptor &source,
    bool hasStat, const Descriptor *errMsg, const char *sourceFile,
    int sourceLine) {
  if (pointer.Elements() == 0) {
    return StatOk;
  }
  int stat{RTNAME(PointerAllocate)(
      pointer, hasStat, errMsg, sourceFile, sourceLine)};
  if (stat == StatOk) {
    Terminator terminator{sourceFile, sourceLine};
    DoFromSourceAssign(pointer, source, terminator);
  }
  return stat;
}

int RTNAME(PointerDeallocate)(Descriptor &pointer, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (!pointer.IsPointer()) {
    return ReturnError(terminator, StatInvalidDescriptor, errMsg, hasStat);
  }
  if (!pointer.IsAllocated()) {
    return ReturnError(terminator, StatBaseNull, errMsg, hasStat);
  }
  return ReturnError(terminator, pointer.Destroy(true, true), errMsg, hasStat);
}

int RTNAME(PointerDeallocatePolymorphic)(Descriptor &pointer,
    const typeInfo::DerivedType *derivedType, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  int stat{RTNAME(PointerDeallocate)(
      pointer, hasStat, errMsg, sourceFile, sourceLine)};
  if (stat == StatOk) {
    DescriptorAddendum *addendum{pointer.Addendum()};
    if (addendum) {
      addendum->set_derivedType(derivedType);
    } else {
      // Unlimited polymorphic descriptors initialized with
      // PointerNullifyIntrinsic do not have an addendum. Make sure the
      // derivedType is null in that case.
      INTERNAL_CHECK(!derivedType);
    }
  }
  return stat;
}

bool RTNAME(PointerIsAssociated)(const Descriptor &pointer) {
  return pointer.raw().base_addr != nullptr;
}

bool RTNAME(PointerIsAssociatedWith)(
    const Descriptor &pointer, const Descriptor *target) {
  if (!target) {
    return pointer.raw().base_addr != nullptr;
  }
  if (!target->raw().base_addr ||
      (target->raw().type != CFI_type_struct && target->ElementBytes() == 0)) {
    return false;
  }
  int rank{pointer.rank()};
  if (pointer.raw().base_addr != target->raw().base_addr ||
      pointer.ElementBytes() != target->ElementBytes() ||
      rank != target->rank()) {
    return false;
  }
  for (int j{0}; j < rank; ++j) {
    const Dimension &pDim{pointer.GetDimension(j)};
    const Dimension &tDim{target->GetDimension(j)};
    auto pExtent{pDim.Extent()};
    if (pExtent == 0 || pExtent != tDim.Extent() ||
        (pExtent != 1 && pDim.ByteStride() != tDim.ByteStride())) {
      return false;
    }
  }
  return true;
}

// TODO: PointerCheckLengthParameter

} // extern "C"
} // namespace Fortran::runtime

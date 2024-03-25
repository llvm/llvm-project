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
#include "environment.h"
#include "stat.h"
#include "terminator.h"
#include "tools.h"
#include "type-info.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(PointerNullifyIntrinsic)(Descriptor &pointer, TypeCategory category,
    int kind, int rank, int corank) {
  INTERNAL_CHECK(corank == 0);
  pointer.Establish(TypeCode{category, kind},
      Descriptor::BytesFor(category, kind), nullptr, rank, nullptr,
      CFI_attribute_pointer);
}

void RTDEF(PointerNullifyCharacter)(Descriptor &pointer, SubscriptValue length,
    int kind, int rank, int corank) {
  INTERNAL_CHECK(corank == 0);
  pointer.Establish(
      kind, length, nullptr, rank, nullptr, CFI_attribute_pointer);
}

void RTDEF(PointerNullifyDerived)(Descriptor &pointer,
    const typeInfo::DerivedType &derivedType, int rank, int corank) {
  INTERNAL_CHECK(corank == 0);
  pointer.Establish(derivedType, nullptr, rank, nullptr, CFI_attribute_pointer);
}

void RTDEF(PointerSetBounds)(Descriptor &pointer, int zeroBasedDim,
    SubscriptValue lower, SubscriptValue upper) {
  INTERNAL_CHECK(zeroBasedDim >= 0 && zeroBasedDim < pointer.rank());
  pointer.GetDimension(zeroBasedDim).SetBounds(lower, upper);
  // The byte strides are computed when the pointer is allocated.
}

// TODO: PointerSetCoBounds

void RTDEF(PointerSetDerivedLength)(
    Descriptor &pointer, int which, SubscriptValue x) {
  DescriptorAddendum *addendum{pointer.Addendum()};
  INTERNAL_CHECK(addendum != nullptr);
  addendum->SetLenParameterValue(which, x);
}

void RTDEF(PointerApplyMold)(
    Descriptor &pointer, const Descriptor &mold, int rank) {
  pointer.ApplyMold(mold, rank);
}

void RTDEF(PointerAssociateScalar)(Descriptor &pointer, void *target) {
  pointer.set_base_addr(target);
}

void RTDEF(PointerAssociate)(Descriptor &pointer, const Descriptor &target) {
  pointer = target;
  pointer.raw().attribute = CFI_attribute_pointer;
}

void RTDEF(PointerAssociateLowerBounds)(Descriptor &pointer,
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

void RTDEF(PointerAssociateRemapping)(Descriptor &pointer,
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

int RTDEF(PointerAllocate)(Descriptor &pointer, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (!pointer.IsPointer()) {
    return ReturnError(terminator, StatInvalidDescriptor, errMsg, hasStat);
  }
  std::size_t elementBytes{pointer.ElementBytes()};
  if (static_cast<std::int64_t>(elementBytes) < 0) {
    // F'2023 7.4.4.2 p5: "If the character length parameter value evaluates
    // to a negative value, the length of character entities declared is zero."
    elementBytes = pointer.raw().elem_len = 0;
  }
  std::size_t byteSize{pointer.Elements() * elementBytes};
  // Add space for a footer to validate during DEALLOCATE.
  constexpr std::size_t align{sizeof(std::uintptr_t)};
  byteSize = ((byteSize + align - 1) / align) * align;
  std::size_t total{byteSize + sizeof(std::uintptr_t)};
  void *p{std::malloc(total)};
  if (!p) {
    return ReturnError(terminator, CFI_ERROR_MEM_ALLOCATION, errMsg, hasStat);
  }
  pointer.set_base_addr(p);
  pointer.SetByteStrides();
  // Fill the footer word with the XOR of the ones' complement of
  // the base address, which is a value that would be highly unlikely
  // to appear accidentally at the right spot.
  std::uintptr_t *footer{
      reinterpret_cast<std::uintptr_t *>(static_cast<char *>(p) + byteSize)};
  *footer = ~reinterpret_cast<std::uintptr_t>(p);
  int stat{StatOk};
  if (const DescriptorAddendum * addendum{pointer.Addendum()}) {
    if (const auto *derived{addendum->derivedType()}) {
      if (!derived->noInitializationNeeded()) {
        stat = Initialize(pointer, *derived, terminator, hasStat, errMsg);
      }
    }
  }
  return ReturnError(terminator, stat, errMsg, hasStat);
}

int RTDEF(PointerAllocateSource)(Descriptor &pointer, const Descriptor &source,
    bool hasStat, const Descriptor *errMsg, const char *sourceFile,
    int sourceLine) {
  int stat{RTNAME(PointerAllocate)(
      pointer, hasStat, errMsg, sourceFile, sourceLine)};
  if (stat == StatOk) {
    Terminator terminator{sourceFile, sourceLine};
    DoFromSourceAssign(pointer, source, terminator);
  }
  return stat;
}

int RTDEF(PointerDeallocate)(Descriptor &pointer, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (!pointer.IsPointer()) {
    return ReturnError(terminator, StatInvalidDescriptor, errMsg, hasStat);
  }
  if (!pointer.IsAllocated()) {
    return ReturnError(terminator, StatBaseNull, errMsg, hasStat);
  }
  if (executionEnvironment.checkPointerDeallocation) {
    // Validate the footer.  This should fail if the pointer doesn't
    // span the entire object, or the object was not allocated as a
    // pointer.
    std::size_t byteSize{pointer.Elements() * pointer.ElementBytes()};
    constexpr std::size_t align{sizeof(std::uintptr_t)};
    byteSize = ((byteSize + align - 1) / align) * align;
    void *p{pointer.raw().base_addr};
    std::uintptr_t *footer{
        reinterpret_cast<std::uintptr_t *>(static_cast<char *>(p) + byteSize)};
    if (*footer != ~reinterpret_cast<std::uintptr_t>(p)) {
      return ReturnError(
          terminator, StatBadPointerDeallocation, errMsg, hasStat);
    }
  }
  return ReturnError(terminator,
      pointer.Destroy(/*finalize=*/true, /*destroyPointers=*/true, &terminator),
      errMsg, hasStat);
}

int RTDEF(PointerDeallocatePolymorphic)(Descriptor &pointer,
    const typeInfo::DerivedType *derivedType, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  int stat{RTNAME(PointerDeallocate)(
      pointer, hasStat, errMsg, sourceFile, sourceLine)};
  if (stat == StatOk) {
    if (DescriptorAddendum * addendum{pointer.Addendum()}) {
      addendum->set_derivedType(derivedType);
      pointer.raw().type = derivedType ? CFI_type_struct : CFI_type_other;
    } else {
      // Unlimited polymorphic descriptors initialized with
      // PointerNullifyIntrinsic do not have an addendum. Make sure the
      // derivedType is null in that case.
      INTERNAL_CHECK(!derivedType);
      pointer.raw().type = CFI_type_other;
    }
  }
  return stat;
}

bool RTDEF(PointerIsAssociated)(const Descriptor &pointer) {
  return pointer.raw().base_addr != nullptr;
}

bool RTDEF(PointerIsAssociatedWith)(
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

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime

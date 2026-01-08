//===-- lib/runtime/pointer.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/pointer.h"
#include "flang-rt/runtime/allocator-registry.h"
#include "flang-rt/runtime/assign-impl.h"
#include "flang-rt/runtime/derived.h"
#include "flang-rt/runtime/environment.h"
#include "flang-rt/runtime/stat.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/tools.h"
#include "flang-rt/runtime/type-info.h"

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

static void RT_API_ATTRS PointerRemapping(Descriptor &pointer,
    const Descriptor &target, const Descriptor &bounds, const char *sourceFile,
    int sourceLine, bool isMonomorphic) {
  Terminator terminator{sourceFile, sourceLine};
  SubscriptValue byteStride{/*captured from first dimension*/};
  std::size_t boundElementBytes{bounds.ElementBytes()};
  std::size_t boundsRank{
      static_cast<std::size_t>(bounds.GetDimension(1).Extent())};
  // We cannot just assign target into pointer descriptor, because
  // the ranks may mismatch. Use target as a mold for initializing
  // the pointer descriptor.
  INTERNAL_CHECK(static_cast<std::size_t>(pointer.rank()) == boundsRank);
  pointer.ApplyMold(target, boundsRank, isMonomorphic);
  pointer.set_base_addr(target.raw().base_addr);
  pointer.raw().attribute = CFI_attribute_pointer;
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
  std::size_t pointerElements{pointer.Elements()};
  std::size_t targetElements{target.Elements()};
  if (pointerElements > targetElements) {
    terminator.Crash("PointerAssociateRemapping: too many elements in remapped "
                     "pointer (%zd > %zd)",
        pointerElements, targetElements);
  }
}

void RTDEF(PointerAssociateRemapping)(Descriptor &pointer,
    const Descriptor &target, const Descriptor &bounds, const char *sourceFile,
    int sourceLine) {
  PointerRemapping(
      pointer, target, bounds, sourceFile, sourceLine, /*isMonomorphic=*/false);
}
void RTDEF(PointerAssociateRemappingMonomorphic)(Descriptor &pointer,
    const Descriptor &target, const Descriptor &bounds, const char *sourceFile,
    int sourceLine) {
  PointerRemapping(
      pointer, target, bounds, sourceFile, sourceLine, /*isMonomorphic=*/true);
}

RT_API_ATTRS void *AllocateValidatedPointerPayload(
    std::size_t byteSize, int allocatorIdx) {
  // Add space for a footer to validate during deallocation.
  constexpr std::size_t align{sizeof(std::uintptr_t)};
  byteSize = ((byteSize + align - 1) / align) * align;
  std::size_t total{byteSize + sizeof(std::uintptr_t)};
  AllocFct alloc{allocatorRegistry.GetAllocator(allocatorIdx)};
  void *p{alloc(total, /*asyncObject=*/nullptr)};
  if (p && allocatorIdx == 0) {
    // Fill the footer word with the XOR of the ones' complement of
    // the base address, which is a value that would be highly unlikely
    // to appear accidentally at the right spot.
    std::uintptr_t *footer{
        reinterpret_cast<std::uintptr_t *>(static_cast<char *>(p) + byteSize)};
    *footer = ~reinterpret_cast<std::uintptr_t>(p);
  }
  return p;
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
  void *p{AllocateValidatedPointerPayload(byteSize, pointer.GetAllocIdx())};
  if (!p) {
    return ReturnError(terminator, CFI_ERROR_MEM_ALLOCATION, errMsg, hasStat);
  }
  pointer.set_base_addr(p);
  pointer.SetByteStrides();
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

static RT_API_ATTRS std::size_t GetByteSize(
    const ISO::CFI_cdesc_t &descriptor) {
  std::size_t rank{descriptor.rank};
  const ISO::CFI_dim_t *dim{descriptor.dim};
  std::size_t byteSize{descriptor.elem_len};
  for (std::size_t j{0}; j < rank; ++j) {
    byteSize *= dim[j].extent;
  }
  return byteSize;
}

bool RT_API_ATTRS ValidatePointerPayload(const ISO::CFI_cdesc_t &desc) {
  std::size_t byteSize{GetByteSize(desc)};
  constexpr std::size_t align{sizeof(std::uintptr_t)};
  byteSize = ((byteSize + align - 1) / align) * align;
  const void *p{desc.base_addr};
  const std::uintptr_t *footer{reinterpret_cast<const std::uintptr_t *>(
      static_cast<const char *>(p) + byteSize)};
  return *footer == ~reinterpret_cast<std::uintptr_t>(p);
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
  if (executionEnvironment.checkPointerDeallocation &&
      pointer.GetAllocIdx() == kDefaultAllocator &&
      !ValidatePointerPayload(pointer.raw())) {
    return ReturnError(terminator, StatBadPointerDeallocation, errMsg, hasStat);
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
  if (!target->raw().base_addr || target->ElementBytes() == 0 ||
      target->Elements() == 0) {
    // F2023, 16.9.20, p5, case (v)-(vi): don't associate pointers with
    // targets that have zero sized storage sequence.
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

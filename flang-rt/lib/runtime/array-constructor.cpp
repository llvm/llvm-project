//===-- lib/runtime/array-constructor.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang-rt/runtime/array-constructor.h"
#include "flang-rt/runtime/derived.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/tools.h"
#include "flang-rt/runtime/type-info.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Runtime/assign.h"

namespace Fortran::runtime {

// Initial allocation size for an array constructor temporary whose extent
// cannot be pre-computed. This could be fined tuned if needed based on actual
// program performance.
//  REAL(4), INTEGER(4), COMPLEX(2), ...   -> 32 elements.
//  REAL(8), INTEGER(8), COMPLEX(4), ...   -> 16 elements.
//  REAL(16), INTEGER(16), COMPLEX(8), ... -> 8 elements.
//  Bigger types -> 4 elements.
static RT_API_ATTRS SubscriptValue initialAllocationSize(
    SubscriptValue initialNumberOfElements, SubscriptValue elementBytes) {
  // Try to guess an optimal initial allocation size in number of elements to
  // avoid doing too many reallocation.
  static constexpr SubscriptValue minNumberOfBytes{128};
  static constexpr SubscriptValue minNumberOfElements{4};
  SubscriptValue numberOfElements{initialNumberOfElements > minNumberOfElements
          ? initialNumberOfElements
          : minNumberOfElements};
  SubscriptValue elementsForMinBytes{minNumberOfBytes / elementBytes};
  return std::max(numberOfElements, elementsForMinBytes);
}

static RT_API_ATTRS void AllocateOrReallocateVectorIfNeeded(
    ArrayConstructorVector &vector, Terminator &terminator,
    SubscriptValue previousToElements, SubscriptValue fromElements) {
  Descriptor &to{vector.to};
  if (to.IsAllocatable() && !to.IsAllocated()) {
    // The descriptor bounds may already be set here if the array constructor
    // extent could be pre-computed, but information about length parameters
    // was missing and required evaluating the first array constructor value.
    if (previousToElements == 0) {
      SubscriptValue allocationSize{
          initialAllocationSize(fromElements, to.ElementBytes())};
      to.GetDimension(0).SetBounds(1, allocationSize);
      RTNAME(AllocatableAllocate)
      (to, /*asyncObject=*/nullptr, /*hasStat=*/false, /*errMsg=*/nullptr,
          vector.sourceFile, vector.sourceLine);
      to.GetDimension(0).SetBounds(1, fromElements);
      vector.actualAllocationSize = allocationSize;
    } else {
      // Do not over-allocate if the final extent was known before pushing the
      // first value: there should be no reallocation.
      RUNTIME_CHECK(terminator, previousToElements >= fromElements);
      RTNAME(AllocatableAllocate)
      (to, /*asyncObject=*/nullptr, /*hasStat=*/false, /*errMsg=*/nullptr,
          vector.sourceFile, vector.sourceLine);
      vector.actualAllocationSize = previousToElements;
    }
  } else {
    SubscriptValue newToElements{vector.nextValuePosition + fromElements};
    if (to.IsAllocatable() && vector.actualAllocationSize < newToElements) {
      // Reallocate. Ensure the current storage is at least doubled to avoid
      // doing too many reallocations.
      SubscriptValue requestedAllocationSize{
          std::max(newToElements, vector.actualAllocationSize * 2)};
      std::size_t newByteSize{requestedAllocationSize * to.ElementBytes()};
      // realloc is undefined with zero new size and ElementBytes() may be null
      // if the character length is null, or if "from" is a zero sized array.
      if (newByteSize > 0) {
        void *p{ReallocateMemoryOrCrash(
            terminator, to.raw().base_addr, newByteSize)};
        to.set_base_addr(p);
      }
      vector.actualAllocationSize = requestedAllocationSize;
      to.GetDimension(0).SetBounds(1, newToElements);
    } else if (previousToElements < newToElements) {
      // Storage is big enough, but descriptor extent must be increased because
      // the final extent was not known before pushing array constructor values.
      to.GetDimension(0).SetBounds(1, newToElements);
    }
  }
}

extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(InitArrayConstructorVector)(ArrayConstructorVector &vector,
    Descriptor &to, bool useValueLengthParameters, const char *sourceFile,
    int sourceLine) {
  Terminator terminator{vector.sourceFile, vector.sourceLine};
  RUNTIME_CHECK(terminator, to.rank() == 1);
  SubscriptValue actualAllocationSize{
      to.IsAllocated() ? static_cast<SubscriptValue>(to.Elements()) : 0};
  (void)new (&vector) ArrayConstructorVector{to, /*nextValuePosition=*/0,
      actualAllocationSize, sourceFile, sourceLine, useValueLengthParameters};
}

void RTDEF(PushArrayConstructorValue)(
    ArrayConstructorVector &vector, const Descriptor &from) {
  Terminator terminator{vector.sourceFile, vector.sourceLine};
  Descriptor &to{vector.to};
  SubscriptValue fromElements{static_cast<SubscriptValue>(from.Elements())};
  SubscriptValue previousToElements{static_cast<SubscriptValue>(to.Elements())};
  if (vector.useValueLengthParameters()) {
    // Array constructor with no type spec.
    if (to.IsAllocatable() && !to.IsAllocated()) {
      // Takes length parameters, if any, from the first value.
      // Note that "to" type must already be set by the caller of this API since
      // it cannot be taken from "from" here: "from" may be polymorphic (have a
      // dynamic type that differs from its declared type) and Fortran 2018 7.8
      // point 4. says that the dynamic type of an array constructor is its
      // declared type: it does not inherit the dynamic type of its ac-value
      // even if if there is no type-spec.
      if (to.type().IsCharacter()) {
        to.raw().elem_len = from.ElementBytes();
      } else if (auto *toAddendum{to.Addendum()}) {
        if (const auto *fromAddendum{from.Addendum()}) {
          if (const auto *toDerived{toAddendum->derivedType()}) {
            std::size_t lenParms{toDerived->LenParameters()};
            for (std::size_t j{0}; j < lenParms; ++j) {
              toAddendum->SetLenParameterValue(
                  j, fromAddendum->LenParameterValue(j));
            }
          }
        }
      }
    } else if (to.type().IsCharacter()) {
      // Fortran 2018 7.8 point 2.
      if (to.ElementBytes() != from.ElementBytes()) {
        terminator.Crash("Array constructor: mismatched character lengths (%d "
                         "!= %d) between "
                         "values of an array constructor without type-spec",
            to.ElementBytes() / to.type().GetCategoryAndKind()->second,
            from.ElementBytes() / from.type().GetCategoryAndKind()->second);
      }
    }
  }
  // Otherwise, the array constructor had a type-spec and the length
  // parameters are already in the "to" descriptor.

  AllocateOrReallocateVectorIfNeeded(
      vector, terminator, previousToElements, fromElements);

  // Create descriptor for "to" element or section being copied to.
  SubscriptValue lower[1]{
      to.GetDimension(0).LowerBound() + vector.nextValuePosition};
  SubscriptValue upper[1]{lower[0] + fromElements - 1};
  SubscriptValue stride[1]{from.rank() == 0 ? 0 : 1};
  StaticDescriptor<maxRank, true, 1> staticDesc;
  Descriptor &toCurrentElement{staticDesc.descriptor()};
  toCurrentElement.EstablishPointerSection(to, lower, upper, stride);
  // Note: toCurrentElement and from have the same number of elements
  // and "toCurrentElement" is not an allocatable so AssignTemporary
  // below works even if "from" rank is bigger than one (and differs
  // from "toCurrentElement") and not time is wasted reshaping
  // "toCurrentElement" to "from" shape.
  RTNAME(AssignTemporary)
  (toCurrentElement, from, vector.sourceFile, vector.sourceLine);
  vector.nextValuePosition += fromElements;
}

void RTDEF(PushArrayConstructorSimpleScalar)(
    ArrayConstructorVector &vector, void *from) {
  Terminator terminator{vector.sourceFile, vector.sourceLine};
  Descriptor &to{vector.to};
  AllocateOrReallocateVectorIfNeeded(vector, terminator, to.Elements(), 1);
  SubscriptValue subscript[1]{
      to.GetDimension(0).LowerBound() + vector.nextValuePosition};
  runtime::memcpy(to.Element<char>(subscript), from, to.ElementBytes());
  ++vector.nextValuePosition;
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime

//===-- runtime/transformational.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transformational.h"
#include "memory.h"
#include "terminator.h"
#include <algorithm>
#include <cinttypes>

namespace Fortran::runtime {

static inline std::int64_t GetInt64(const char *p, std::size_t bytes) {
  switch (bytes) {
  case 1:
    return *reinterpret_cast<const std::int8_t *>(p);
  case 2:
    return *reinterpret_cast<const std::int16_t *>(p);
  case 4:
    return *reinterpret_cast<const std::int32_t *>(p);
  case 8:
    return *reinterpret_cast<const std::int64_t *>(p);
  default:
    Terminator terminator{__FILE__, __LINE__};
    terminator.Crash("no case for %dz bytes", bytes);
  }
}

// F2018 16.9.163
OwningPtr<Descriptor> RESHAPE(const Descriptor &source, const Descriptor &shape,
    const Descriptor *pad, const Descriptor *order) {
  // Compute and check the rank of the result.
  Terminator terminator{__FILE__, __LINE__};
  RUNTIME_CHECK(terminator, shape.rank() == 1);
  RUNTIME_CHECK(terminator, shape.type().IsInteger());
  SubscriptValue resultRank{shape.GetDimension(0).Extent()};
  RUNTIME_CHECK(terminator,
      resultRank >= 0 && resultRank <= static_cast<SubscriptValue>(maxRank));

  // Extract and check the shape of the result; compute its element count.
  SubscriptValue lowerBound[maxRank]; // all 1's
  SubscriptValue resultExtent[maxRank];
  std::size_t shapeElementBytes{shape.ElementBytes()};
  std::size_t resultElements{1};
  SubscriptValue shapeSubscript{shape.GetDimension(0).LowerBound()};
  for (SubscriptValue j{0}; j < resultRank; ++j, ++shapeSubscript) {
    lowerBound[j] = 1;
    resultExtent[j] =
        GetInt64(shape.Element<char>(&shapeSubscript), shapeElementBytes);
    RUNTIME_CHECK(terminator, resultExtent[j] >= 0);
    resultElements *= resultExtent[j];
  }

  // Check that there are sufficient elements in the SOURCE=, or that
  // the optional PAD= argument is present and nonempty.
  std::size_t elementBytes{source.ElementBytes()};
  std::size_t sourceElements{source.Elements()};
  std::size_t padElements{pad ? pad->Elements() : 0};
  if (resultElements < sourceElements) {
    RUNTIME_CHECK(terminator, padElements > 0);
    RUNTIME_CHECK(terminator, pad->ElementBytes() == elementBytes);
  }

  // Extract and check the optional ORDER= argument, which must be a
  // permutation of [1..resultRank].
  int dimOrder[maxRank];
  if (order) {
    RUNTIME_CHECK(terminator, order->rank() == 1);
    RUNTIME_CHECK(terminator, order->type().IsInteger());
    RUNTIME_CHECK(terminator, order->GetDimension(0).Extent() == resultRank);
    std::uint64_t values{0};
    SubscriptValue orderSubscript{order->GetDimension(0).LowerBound()};
    for (SubscriptValue j{0}; j < resultRank; ++j, ++orderSubscript) {
      auto k{GetInt64(
          order->OffsetElement<char>(orderSubscript), shapeElementBytes)};
      RUNTIME_CHECK(
          terminator, k >= 1 && k <= resultRank && !((values >> k) & 1));
      values |= std::uint64_t{1} << k;
      dimOrder[k - 1] = j;
    }
  } else {
    for (int j{0}; j < resultRank; ++j) {
      dimOrder[j] = j;
    }
  }

  // Create and populate the result's descriptor.
  const DescriptorAddendum *sourceAddendum{source.Addendum()};
  const typeInfo::DerivedType *sourceDerivedType{
      sourceAddendum ? sourceAddendum->derivedType() : nullptr};
  OwningPtr<Descriptor> result;
  if (sourceDerivedType) {
    result = Descriptor::Create(*sourceDerivedType, nullptr, resultRank,
        resultExtent, CFI_attribute_allocatable);
  } else {
    result = Descriptor::Create(source.type(), elementBytes, nullptr,
        resultRank, resultExtent,
        CFI_attribute_allocatable); // TODO rearrange these arguments
  }
  DescriptorAddendum *resultAddendum{result->Addendum()};
  RUNTIME_CHECK(terminator, resultAddendum);
  resultAddendum->flags() |= DescriptorAddendum::DoNotFinalize;
  if (sourceDerivedType) {
    std::size_t lenParameters{sourceAddendum->LenParameters()};
    for (std::size_t j{0}; j < lenParameters; ++j) {
      resultAddendum->SetLenParameterValue(
          j, sourceAddendum->LenParameterValue(j));
    }
  }
  // Allocate storage for the result's data.
  int status{result->Allocate(lowerBound, resultExtent)};
  if (status != CFI_SUCCESS) {
    terminator.Crash("RESHAPE: Allocate failed (error %d)", status);
  }

  // Populate the result's elements.
  SubscriptValue resultSubscript[maxRank];
  result->GetLowerBounds(resultSubscript);
  SubscriptValue sourceSubscript[maxRank];
  source.GetLowerBounds(sourceSubscript);
  std::size_t resultElement{0};
  std::size_t elementsFromSource{std::min(resultElements, sourceElements)};
  for (; resultElement < elementsFromSource; ++resultElement) {
    std::memcpy(result->Element<void>(resultSubscript),
        source.Element<const void>(sourceSubscript), elementBytes);
    source.IncrementSubscripts(sourceSubscript);
    result->IncrementSubscripts(resultSubscript, dimOrder);
  }
  if (resultElement < resultElements) {
    // Remaining elements come from the optional PAD= argument.
    SubscriptValue padSubscript[maxRank];
    pad->GetLowerBounds(padSubscript);
    for (; resultElement < resultElements; ++resultElement) {
      std::memcpy(result->Element<void>(resultSubscript),
          pad->Element<const void>(padSubscript), elementBytes);
      pad->IncrementSubscripts(padSubscript);
      result->IncrementSubscripts(resultSubscript, dimOrder);
    }
  }

  return result;
}
} // namespace Fortran::runtime

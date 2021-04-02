//===-- runtime/misc-intrinsic.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "misc-intrinsic.h"
#include "descriptor.h"
#include "terminator.h"
#include <algorithm>
#include <cstring>

namespace Fortran::runtime {
extern "C" {

void RTNAME(Transfer)(Descriptor &result, const Descriptor &source,
    const Descriptor &mold, const char *sourceFile, int line) {
  if (mold.rank() > 0) {
    std::size_t moldElementBytes{mold.ElementBytes()};
    std::size_t elements{
        (source.Elements() * source.ElementBytes() + moldElementBytes - 1) /
        moldElementBytes};
    return RTNAME(TransferSize)(result, source, mold, sourceFile, line,
        static_cast<std::int64_t>(elements));
  } else {
    return RTNAME(TransferSize)(result, source, mold, sourceFile, line, 1);
  }
}

void RTNAME(TransferSize)(Descriptor &result, const Descriptor &source,
    const Descriptor &mold, const char *sourceFile, int line,
    std::int64_t size) {
  int rank{mold.rank() > 0 ? 1 : 0};
  std::size_t elementBytes{mold.ElementBytes()};
  result.Establish(mold.type(), elementBytes, nullptr, rank, nullptr,
      CFI_attribute_allocatable, mold.Addendum() != nullptr);
  if (rank > 0) {
    result.GetDimension(0).SetBounds(1, size);
  }
  if (const DescriptorAddendum * addendum{mold.Addendum()}) {
    *result.Addendum() = *addendum;
    auto &flags{result.Addendum()->flags()};
    flags &= ~DescriptorAddendum::StaticDescriptor;
    flags |= DescriptorAddendum::DoNotFinalize;
  }
  if (int stat{result.Allocate()}) {
    Terminator{sourceFile, line}.Crash(
        "TRANSFER: could not allocate memory for result; STAT=%d", stat);
  }
  char *to{result.OffsetElement<char>()};
  std::size_t resultBytes{size * elementBytes};
  const std::size_t sourceElementBytes{source.ElementBytes()};
  std::size_t sourceElements{source.Elements()};
  SubscriptValue sourceAt[maxRank];
  source.GetLowerBounds(sourceAt);
  while (resultBytes > 0 && sourceElements > 0) {
    std::size_t toMove{std::min(resultBytes, sourceElementBytes)};
    std::memcpy(to, source.Element<char>(sourceAt), toMove);
    to += toMove;
    resultBytes -= toMove;
    --sourceElements;
    source.IncrementSubscripts(sourceAt);
  }
  if (resultBytes > 0) {
    std::memset(to, 0, resultBytes);
  }
}

} // extern "C"
} // namespace Fortran::runtime

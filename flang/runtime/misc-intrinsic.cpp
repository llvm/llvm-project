//===-- runtime/misc-intrinsic.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/misc-intrinsic.h"
#include "terminator.h"
#include "flang/Runtime/descriptor.h"
#include <algorithm>
#include <cstring>
#include <optional>

namespace Fortran::runtime {

static void TransferImpl(Descriptor &result, const Descriptor &source,
    const Descriptor &mold, const char *sourceFile, int line,
    std::optional<std::int64_t> resultExtent) {
  int rank{resultExtent.has_value() ? 1 : 0};
  std::size_t elementBytes{mold.ElementBytes()};
  result.Establish(mold.type(), elementBytes, nullptr, rank, nullptr,
      CFI_attribute_allocatable, mold.Addendum() != nullptr);
  if (resultExtent) {
    result.GetDimension(0).SetBounds(1, *resultExtent);
  }
  if (const DescriptorAddendum * addendum{mold.Addendum()}) {
    *result.Addendum() = *addendum;
  }
  if (int stat{result.Allocate()}) {
    Terminator{sourceFile, line}.Crash(
        "TRANSFER: could not allocate memory for result; STAT=%d", stat);
  }
  char *to{result.OffsetElement<char>()};
  std::size_t resultBytes{result.Elements() * result.ElementBytes()};
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

extern "C" {

void RTNAME(Transfer)(Descriptor &result, const Descriptor &source,
    const Descriptor &mold, const char *sourceFile, int line) {
  if (mold.rank() > 0) {
    std::size_t moldElementBytes{mold.ElementBytes()};
    std::size_t elements{
        (source.Elements() * source.ElementBytes() + moldElementBytes - 1) /
        moldElementBytes};
    return TransferImpl(result, source, mold, sourceFile, line,
        static_cast<std::int64_t>(elements));
  } else {
    return TransferImpl(result, source, mold, sourceFile, line, {});
  }
}

void RTNAME(TransferSize)(Descriptor &result, const Descriptor &source,
    const Descriptor &mold, const char *sourceFile, int line,
    std::int64_t size) {
  return TransferImpl(result, source, mold, sourceFile, line, size);
}

} // extern "C"
} // namespace Fortran::runtime

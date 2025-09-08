//===-- lib/runtime/misc-intrinsic.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/misc-intrinsic.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/tools.h"
#include "flang/Common/optional.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace Fortran::runtime {

static RT_API_ATTRS void TransferImpl(Descriptor &result,
    const Descriptor &source, const Descriptor &mold, const char *sourceFile,
    int line, common::optional<std::int64_t> resultExtent) {
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
  if (int stat{result.Allocate(kNoAsyncObject)}) {
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
RT_EXT_API_GROUP_BEGIN

void RTDEF(Rename)(const Descriptor &path1, const Descriptor &path2,
    const Descriptor *status, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};
#if !defined(RT_DEVICE_COMPILATION)
  char *pathSrc{EnsureNullTerminated(
      path1.OffsetElement(), path1.ElementBytes(), terminator)};
  char *pathDst{EnsureNullTerminated(
      path2.OffsetElement(), path2.ElementBytes(), terminator)};

  // We simply call rename(2) from POSIX
  int result{rename(pathSrc, pathDst)};
  if (status) {
    // When an error has happened,
    int errorCode{0}; // Assume success
    if (result != 0) {
      // The rename operation has failed, so return the error code as status.
      errorCode = errno;
    }
    StoreIntToDescriptor(status, errorCode, terminator);
  }

  // Deallocate memory if EnsureNullTerminated dynamically allocated memory
  if (pathSrc != path1.OffsetElement()) {
    FreeMemory(pathSrc);
  }
  if (pathDst != path2.OffsetElement()) {
    FreeMemory(pathDst);
  }
#else // !defined(RT_DEVICE_COMPILATION)
  terminator.Crash("RENAME intrinsic is only supported on host devices");
#endif // !defined(RT_DEVICE_COMPILATION)
}

void RTDEF(Transfer)(Descriptor &result, const Descriptor &source,
    const Descriptor &mold, const char *sourceFile, int line) {
  common::optional<std::int64_t> elements;
  if (mold.rank() > 0) {
    if (std::size_t sourceElementBytes{
            source.Elements() * source.ElementBytes()}) {
      if (std::size_t moldElementBytes{mold.ElementBytes()}) {
        elements = static_cast<std::int64_t>(
            (sourceElementBytes + moldElementBytes - 1) / moldElementBytes);
      } else {
        Terminator{sourceFile, line}.Crash("TRANSFER: zero-sized type of MOLD= "
                                           "when SOURCE= is not zero-sized");
      }
    } else {
      elements = std::int64_t{0};
    }
  }
  return TransferImpl(
      result, source, mold, sourceFile, line, std::move(elements));
}

void RTDEF(TransferSize)(Descriptor &result, const Descriptor &source,
    const Descriptor &mold, const char *sourceFile, int line,
    std::int64_t size) {
  return TransferImpl(result, source, mold, sourceFile, line, size);
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime

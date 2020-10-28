//===-- utilities.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/utilities.h"

#include <assert.h>

namespace gwp_asan {
// See `bionic/tests/malloc_test.cpp` in the Android source for documentation
// regarding their alignment guarantees. We always round up to the closest
// 8-byte window. As GWP-ASan's malloc(X) can always get exactly an X-sized
// allocation, an allocation that rounds up to 16-bytes will always be given a
// 16-byte aligned allocation.
static size_t alignBionic(size_t RealAllocationSize) {
  if (RealAllocationSize % 8 == 0)
    return RealAllocationSize;
  return RealAllocationSize + 8 - (RealAllocationSize % 8);
}

static size_t alignPowerOfTwo(size_t RealAllocationSize) {
  if (RealAllocationSize <= 2)
    return RealAllocationSize;
  if (RealAllocationSize <= 4)
    return 4;
  if (RealAllocationSize <= 8)
    return 8;
  if (RealAllocationSize % 16 == 0)
    return RealAllocationSize;
  return RealAllocationSize + 16 - (RealAllocationSize % 16);
}

#ifdef __BIONIC__
static constexpr AlignmentStrategy PlatformDefaultAlignment =
    AlignmentStrategy::BIONIC;
#else // __BIONIC__
static constexpr AlignmentStrategy PlatformDefaultAlignment =
    AlignmentStrategy::POWER_OF_TWO;
#endif // __BIONIC__

size_t rightAlignedAllocationSize(size_t RealAllocationSize,
                                  AlignmentStrategy Align) {
  assert(RealAllocationSize > 0);
  if (Align == AlignmentStrategy::DEFAULT)
    Align = PlatformDefaultAlignment;

  switch (Align) {
  case AlignmentStrategy::BIONIC:
    return alignBionic(RealAllocationSize);
  case AlignmentStrategy::POWER_OF_TWO:
    return alignPowerOfTwo(RealAllocationSize);
  case AlignmentStrategy::PERFECT:
    return RealAllocationSize;
  case AlignmentStrategy::DEFAULT:
    __builtin_unreachable();
  }
  __builtin_unreachable();
}
} // namespace gwp_asan

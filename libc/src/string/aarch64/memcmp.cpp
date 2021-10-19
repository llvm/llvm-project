//===-- Implementation of memcmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memcmp.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/elements.h"
#include <stddef.h> // size_t

namespace __llvm_libc {

static int memcmp_aarch64(const char *lhs, const char *rhs, size_t count) {
  // Use aarch64 strategies (_1, _2, _3 ...)
  using namespace __llvm_libc::aarch64;

  if (count == 0) // [0, 0]
    return 0;
  if (count == 1) // [1, 1]
    return ThreeWayCompare<_1>(lhs, rhs);
  if (count == 2) // [2, 2]
    return ThreeWayCompare<_2>(lhs, rhs);
  if (count == 3) // [3, 3]
    return ThreeWayCompare<_3>(lhs, rhs);
  if (count < 8) // [4, 7]
    return ThreeWayCompare<HeadTail<_4>>(lhs, rhs, count);
  if (count < 16) // [8, 15]
    return ThreeWayCompare<HeadTail<_8>>(lhs, rhs, count);
  if (unlikely(count >= 128)) // [128, ∞]
    return ThreeWayCompare<Align<_16>::Then<Loop<_32>>>(lhs, rhs, count);
  if (!Equals<_16>(lhs, rhs)) // [16, 16]
    return ThreeWayCompare<_16>(lhs, rhs);
  if (count < 32) // [17, 31]
    return ThreeWayCompare<Tail<_16>>(lhs, rhs, count);
  if (!Equals<Skip<16>::Then<_16>>(lhs, rhs)) // [32, 32]
    return ThreeWayCompare<Skip<16>::Then<_16>>(lhs, rhs);
  if (count < 64) // [33, 63]
    return ThreeWayCompare<Tail<_32>>(lhs, rhs, count);
  // [64, 127]
  return ThreeWayCompare<Skip<32>::Then<Loop<_16>>>(lhs, rhs, count);
}

LLVM_LIBC_FUNCTION(int, memcmp,
                   (const void *lhs, const void *rhs, size_t count)) {
  return memcmp_aarch64(reinterpret_cast<const char *>(lhs),
                        reinterpret_cast<const char *>(rhs), count);
}

} // namespace __llvm_libc

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
namespace aarch64 {

static int memcmp_impl(const char *lhs, const char *rhs, size_t count) {
  if (count == 0)
    return 0;
  if (count == 1)
    return ThreeWayCompare<_1>(lhs, rhs);
  else if (count == 2)
    return ThreeWayCompare<_2>(lhs, rhs);
  else if (count == 3)
    return ThreeWayCompare<_3>(lhs, rhs);
  else if (count < 8)
    return ThreeWayCompare<HeadTail<_4>>(lhs, rhs, count);
  else if (count < 16)
    return ThreeWayCompare<HeadTail<_8>>(lhs, rhs, count);
  else if (count < 128) {
    if (Equals<_16>(lhs, rhs)) {
      if (count < 32)
        return ThreeWayCompare<Tail<_16>>(lhs, rhs, count);
      else {
        if (Equals<_16>(lhs + 16, rhs + 16)) {
          if (count < 64)
            return ThreeWayCompare<Tail<_32>>(lhs, rhs, count);
          if (count < 128)
            return ThreeWayCompare<Loop<_16>>(lhs + 32, rhs + 32, count - 32);
        } else
          return ThreeWayCompare<_16>(lhs + count - 32, rhs + count - 32);
      }
    }
    return ThreeWayCompare<_16>(lhs, rhs);
  } else
    return ThreeWayCompare<Align<_16, Arg::Lhs>::Then<Loop<_32>>>(lhs, rhs,
                                                                  count);
}
} // namespace aarch64

LLVM_LIBC_FUNCTION(int, memcmp,
                   (const void *lhs, const void *rhs, size_t count)) {

  const char *_lhs = reinterpret_cast<const char *>(lhs);
  const char *_rhs = reinterpret_cast<const char *>(rhs);
  return aarch64::memcmp_impl(_lhs, _rhs, count);
}

} // namespace __llvm_libc

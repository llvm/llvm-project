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

static int memcmp_impl(const char *lhs, const char *rhs, size_t count) {
#if defined(__i386__) || defined(__x86_64__)
  using namespace ::__llvm_libc::x86;
#else
  using namespace ::__llvm_libc::scalar;
#endif

  if (count == 0)
    return 0;
  if (count == 1)
    return ThreeWayCompare<_1>(lhs, rhs);
  if (count == 2)
    return ThreeWayCompare<_2>(lhs, rhs);
  if (count == 3)
    return ThreeWayCompare<_3>(lhs, rhs);
  if (count <= 8)
    return ThreeWayCompare<HeadTail<_4>>(lhs, rhs, count);
  if (count <= 16)
    return ThreeWayCompare<HeadTail<_8>>(lhs, rhs, count);
  if (count <= 32)
    return ThreeWayCompare<HeadTail<_16>>(lhs, rhs, count);
  if (count <= 64)
    return ThreeWayCompare<HeadTail<_32>>(lhs, rhs, count);
  if (count <= 128)
    return ThreeWayCompare<HeadTail<_64>>(lhs, rhs, count);
  return ThreeWayCompare<Align<_32>::Then<Loop<_32>>>(lhs, rhs, count);
}

LLVM_LIBC_FUNCTION(int, memcmp,
                   (const void *lhs, const void *rhs, size_t count)) {
  return memcmp_impl(static_cast<const char *>(lhs),
                     static_cast<const char *>(rhs), count);
}

} // namespace __llvm_libc

//===-- Implementation of memmove -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memmove.h"

#include "src/__support/common.h"
#include "src/__support/integer_operations.h"
#include "src/string/memcpy.h"
#include <stddef.h> // size_t, ptrdiff_t

namespace __llvm_libc {

static inline void move_byte_backward(char *dst, const char *src,
                                      size_t count) {
  for (size_t offset = count - 1; count; --count, --offset)
    dst[offset] = src[offset];
}

static void memmove_impl(char *dst, const char *src, size_t count) {

  // If the distance between `src` and `dst` is equal to or greater
  // than count (integerAbs(src - dst) >= count), they would not overlap.
  // e.g. greater     equal       overlapping
  //      [12345678]  [12345678]  [12345678]
  // src: [_ab_____]  [_ab_____]  [_ab_____]
  // dst: [_____yz_]  [___yz___]  [__yz____]

  // Call `memcpy` when `src` and `dst` do not overlap.
  if (__llvm_libc::integerAbs(src - dst) >= static_cast<ptrdiff_t>(count))
    __llvm_libc::memcpy(dst, src, count);

  // Overlap cases.
  // If `dst` starts before `src` (dst < src), copy forward from beginning to
  // end. If `dst` starts after `src` (dst > src), copy backward from end to
  // beginning. If `dst` and `src` start at the same address (dst == src), do
  // nothing.
  // e.g. forward      backward
  //             *->    <-*
  // src: [___abcde_]  [_abcde___]
  // dst: [_abc--___]  [___--cde_]

  // In `memcpy` implementation, it copies bytes forward from beginning to
  // end. Otherwise, `memmove` unit tests will break.
  if (dst < src)
    __llvm_libc::memcpy(dst, src, count);

  // TODO: Optimize `move_byte_xxx(...)` functions.
  if (dst > src)
    move_byte_backward(dst, src, count);
}

LLVM_LIBC_FUNCTION(void *, memmove,
                   (void *dst, const void *src, size_t count)) {
  memmove_impl(reinterpret_cast<char *>(dst),
               reinterpret_cast<const char *>(src), count);
  return dst;
}

} // namespace __llvm_libc

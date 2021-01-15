//===-- Implementation of memmove -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memmove.h"
#include "src/__support/common.h"
#include "src/stdlib/abs_utils.h"
#include "src/string/memcpy.h"
#include <stddef.h> // size_t, ptrdiff_t
#include <unistd.h> // ssize_t

namespace __llvm_libc {

// src_m and dest_m might be the beginning or end.
static inline void move_byte(unsigned char *dest_m, const unsigned char *src_m,
                             size_t count, ssize_t direction) {
  for (ssize_t offset = 0; count; --count, offset += direction)
    dest_m[offset] = src_m[offset];
}

LLVM_LIBC_FUNCTION(void *, memmove,
                   (void *dest, const void *src, size_t count)) {
  unsigned char *dest_c = reinterpret_cast<unsigned char *>(dest);
  const unsigned char *src_c = reinterpret_cast<const unsigned char *>(src);

  // If the distance between src_c and dest_c is equal to or greater
  // than count (integer_abs(src_c - dest_c) >= count), they would not overlap.
  // e.g.   greater     equal       overlapping
  //        [12345678]  [12345678]  [12345678]
  // src_c: [_ab_____]  [_ab_____]  [_ab_____]
  // dest_c:[_____yz_]  [___yz___]  [__yz____]

  // Use memcpy if src_c and dest_c do not overlap.
  if (__llvm_libc::integer_abs(src_c - dest_c) >= static_cast<ptrdiff_t>(count))
    return __llvm_libc::memcpy(dest_c, src_c, count);

  // Overlap cases.
  // If dest_c starts before src_c (dest_c < src_c), copy forward(pointer add 1)
  // from beginning to end.
  // If dest_c starts after src_c (dest_c > src_c), copy backward(pointer add
  // -1) from end to beginning.
  // If dest_c and src_c start at the same address (dest_c == src_c),
  // just return dest.
  // e.g.    forward      backward
  //             *-->        <--*
  // src_c : [___abcde_]  [_abcde___]
  // dest_c: [_abc--___]  [___--cde_]

  // TODO: Optimize `move_byte(...)` function.
  if (dest_c < src_c)
    move_byte(dest_c, src_c, count, /*pointer add*/ 1);
  if (dest_c > src_c)
    move_byte(dest_c + count - 1, src_c + count - 1, count, /*pointer add*/ -1);
  return dest;
}

} // namespace __llvm_libc

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

static inline void move_byte_forward(char *dest_m, const char *src_m,
                                     size_t count) {
  for (size_t offset = 0; count; --count, ++offset)
    dest_m[offset] = src_m[offset];
}

static inline void move_byte_backward(char *dest_m, const char *src_m,
                                      size_t count) {
  for (size_t offset = count - 1; count; --count, --offset)
    dest_m[offset] = src_m[offset];
}

LLVM_LIBC_FUNCTION(void *, memmove,
                   (void *dest, const void *src, size_t count)) {
  char *dest_c = reinterpret_cast<char *>(dest);
  const char *src_c = reinterpret_cast<const char *>(src);

  // If the distance between src_c and dest_c is equal to or greater
  // than count (integerAbs(src_c - dest_c) >= count), they would not overlap.
  // e.g.   greater     equal       overlapping
  //        [12345678]  [12345678]  [12345678]
  // src_c: [_ab_____]  [_ab_____]  [_ab_____]
  // dest_c:[_____yz_]  [___yz___]  [__yz____]

  // Use memcpy if src_c and dest_c do not overlap.
  if (__llvm_libc::integerAbs(src_c - dest_c) >= static_cast<ptrdiff_t>(count))
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

  // TODO: Optimize `move_byte_xxx(...)` functions.
  if (dest_c < src_c)
    move_byte_forward(dest_c, src_c, count);
  if (dest_c > src_c)
    move_byte_backward(dest_c, src_c, count);
  return dest;
}

} // namespace __llvm_libc

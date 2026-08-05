//===-- str{,case}cmp implementation ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_STRCMP_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_STRCMP_H

#include "src/__support/macros/attributes.h"   // LIBC_INLINE
#include "src/__support/macros/config.h"       // LIBC_NAMESPACE_DECL
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include <stddef.h>
#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE uint64_t is_null_terminated(uint64_t v) {
  return (v - 0x0101010101010101ULL) & ~v & 0x8080808080808080ULL;
}

LIBC_INLINE uint64_t load(const char *ptr) {
  uint64_t val{0};
  __builtin_memcpy(&val, ptr, sizeof(uint64_t));
  return val;
}

template <typename Comp>
LIBC_INLINE constexpr int inline_strcmp(const char *left, const char *right,
                                        Comp &&comp) {
  // Page boundry check fallback to generic version
  if (LIBC_UNLIKELY((reinterpret_cast<uintptr_t>(left) & 4095) > 4088 ||
                    (reinterpret_cast<uintptr_t>(right) & 4095) > 4088)) {
    for (; *left && !comp(*left, *right); ++left, ++right)
      ;
    return comp(static_cast<unsigned char>(*left),
                static_cast<unsigned char>(*right));
  }
  while (1) {
    uint64_t val1 = load(left);
    uint64_t val2 = load(right);
    uint64_t diff = val1 ^ val2;
    uint64_t null_mask = is_null_terminated(val1);
    // Check for character mismatch or null terminator
    uint64_t zero_or_diff = diff | null_mask;
    if (zero_or_diff != 0) {
      size_t byte_pos = __builtin_ctzll(zero_or_diff) >> 3;
      unsigned char c1 = static_cast<unsigned char>(left[byte_pos]);
      unsigned char c2 = static_cast<unsigned char>(right[byte_pos]);
      return comp(c1, c2);
    }
    left += 8;
    right += 8;
  }
}

template <typename Comp>
LIBC_INLINE constexpr int inline_strncmp(const char *left, const char *right,
                                         size_t n, Comp &&comp) {
  if (n == 0)
    return 0;

  if (LIBC_UNLIKELY((reinterpret_cast<uintptr_t>(left) & 4095) > 4088 ||
                    (reinterpret_cast<uintptr_t>(right) & 4095) > 4088)) {
    for (; n > 1; --n, ++left, ++right) {
      char lc = *left;
      if (!comp(lc, '\0') || comp(lc, *right))
        break;
    }
    return comp(static_cast<unsigned char>(*left),
                static_cast<unsigned char>(*right));
  }

  for (; n >= 8; n -= 8, left += 8, right += 8) {
    uint64_t val1 = load(left);
    uint64_t val2 = load(right);
    uint64_t diff = val1 ^ val2;
    uint64_t null_mask = is_null_terminated(val1);

    uint64_t zero_or_diff = diff | null_mask;
    if (zero_or_diff != 0) {
      size_t byte_pos = __builtin_ctzll(zero_or_diff) >> 3;
      // If the difference happens past 'n' remaining bytes, they are equal up
      // to n
      if (byte_pos >= n)
        return 0;
      unsigned char c1 = static_cast<unsigned char>(left[byte_pos]);
      unsigned char c2 = static_cast<unsigned char>(right[byte_pos]);
      return comp(c1, c2);
    }
  }
  // Handle remaining 8 bytes if not found in the first loop
  for (; n > 1; n--, ++left, ++right) {
    char lc = *left;
    if (!comp(lc, '\0') || comp(lc, *right))
      break;
  }
  return comp(static_cast<unsigned char>(*left),
              static_cast<unsigned char>(*right));
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_STRCMP_H

//===-- memmem implementation -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMMEM_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMMEM_IMPLEMENTATIONS_H

#include <stddef.h>

namespace __llvm_libc {

template <typename Comp>
constexpr static void *
memmem_implementation(const void *haystack, size_t haystack_len,
                      const void *needle, size_t needle_len, Comp &&comp) {
  // TODO: simple brute force implementation. This can be
  // improved upon using well known string matching algorithms.
  if (!needle_len)
    return const_cast<void *>(haystack);

  if (needle_len > haystack_len)
    return nullptr;

  const unsigned char *h = static_cast<const unsigned char *>(haystack);
  const unsigned char *n = static_cast<const unsigned char *>(needle);
  for (size_t i = 0; i <= (haystack_len - needle_len); ++i) {
    size_t j = 0;
    for (; j < needle_len && !comp(h[i + j], n[j]); ++j)
      ;
    if (j == needle_len)
      return const_cast<unsigned char *>(h + i);
  }
  return nullptr;
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMMEM_IMPLEMENTATIONS_H

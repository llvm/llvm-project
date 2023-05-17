//===-- str{,case}str implementation ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_STRSTR_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_STRSTR_IMPLEMENTATIONS_H

#include "src/string/memory_utils/memmem_implementations.h"
#include "src/string/string_utils.h"
#include <stddef.h>

namespace __llvm_libc {

template <typename Comp>
constexpr static char *strstr_implementation(const char *haystack,
                                             const char *needle, Comp &&comp) {
  void *result = memmem_implementation(
      static_cast<const void *>(haystack), internal::string_length(haystack),
      static_cast<const void *>(needle), internal::string_length(needle), comp);
  return static_cast<char *>(result);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_STRSTR_IMPLEMENTATIONS_H

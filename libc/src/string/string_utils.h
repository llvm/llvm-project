//===-- String utils --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_SRC_STRING_STRING_UTILS_H
#define LIBC_SRC_STRING_STRING_UTILS_H

#include "src/string/memory_utils/utils.h"

#include "utils/CPP/Bitset.h"
#include <stddef.h> // size_t

namespace __llvm_libc {
namespace internal {

// Returns the maximum length span that contains only characters not found in
// 'segment'. If no characters are found, returns the length of 'src'.
static inline size_t complementary_span(const char *src, const char *segment) {
  const char *initial = src;
  cpp::Bitset<256> bitset;

  for (; *segment; ++segment)
    bitset.set(*segment);
  for (; *src && !bitset.test(*src); ++src)
    ;
  return src - initial;
}

} // namespace internal
} // namespace __llvm_libc

#endif //  LIBC_SRC_STRING_STRING_UTILS_H

//===-- String utils --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_SRC_STRING_STRING_UTILS_H
#define LIBC_SRC_STRING_STRING_UTILS_H

#include "utils/CPP/Bitset.h"
#include <stddef.h> // size_t

namespace __llvm_libc {
namespace internal {

// Returns the length of a string, denoted by the first occurrence
// of a null terminator.
static inline size_t string_length(const char *src) {
  size_t length;
  for (length = 0; *src; ++src, ++length)
    ;
  return length;
}

// Returns the first occurrence of 'ch' within the first 'n' characters of
// 'src'. If 'ch' is not found, returns nullptr.
static inline void *find_first_character(const unsigned char *src,
                                         unsigned char ch, size_t n) {
  for (; n && *src != ch; --n, ++src)
    ;
  return n ? const_cast<unsigned char *>(src) : nullptr;
}

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

// Given the similarities between strtok and strtok_r, we can implement both
// using a utility function. On the first call, 'src' is scanned for the
// first character not found in 'delimiter_string'. Once found, it scans until
// the first character in the 'delimiter_string' or the null terminator is
// found. We define this span as a token. The end of the token is appended with
// a null terminator, and the token is returned. The point where the last token
// is found is then stored within 'context' for subsequent calls. Subsequent
// calls will use 'context' when a nullptr is passed in for 'src'. Once the null
// terminating character is reached, returns a nullptr.
static inline char *string_token(char *__restrict src,
                                 const char *__restrict delimiter_string,
                                 char **__restrict saveptr) {
  cpp::Bitset<256> delimiter_set;
  for (; *delimiter_string; ++delimiter_string)
    delimiter_set.set(*delimiter_string);

  src = src ? src : *saveptr;
  for (; *src && delimiter_set.test(*src); ++src)
    ;
  if (!*src) {
    *saveptr = src;
    return nullptr;
  }
  char *token = src;
  for (; *src && !delimiter_set.test(*src); ++src)
    ;
  if (*src) {
    *src = '\0';
    ++src;
  }
  *saveptr = src;
  return token;
}

} // namespace internal
} // namespace __llvm_libc

#endif //  LIBC_SRC_STRING_STRING_UTILS_H

//===-- Stdlib utils --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_SRC_STDLIB_STDLIB_UTILS_H
#define LIBC_SRC_STDLIB_STDLIB_UTILS_H

#include "src/__support/ctype_utils.h"
#include <errno.h>
#include <limits.h>

namespace __llvm_libc {
namespace internal {

// Returns a pointer to the first character in src that is not a whitespace
// character (as determined by isspace())
static inline const char *first_non_whitespace(const char *__restrict src) {
  while (internal::isspace(*src)) {
    ++src;
  }
  return src;
}

static inline int b36_char_to_int(char input) {
  if (isdigit(input))
    return input - '0';
  if (isalpha(input))
    return (input | 32) + 10 - 'a';
  return 0;
}

// Takes the address of the string pointer and parses the base from the start of
// it. This will advance the string pointer.
static inline int infer_base(const char *__restrict *__restrict src) {
  if (**src == '0') {
    ++(*src);
    if ((**src | 32) == 'x') {
      ++(*src);
      return 16;
    }
    return 8;
  }
  return 10;
}

// Takes a pointer to a string, a pointer to a string pointer, and the base to
// convert to. This function is used as the backend for all of the string to int
// functions.
static inline long long strtoll(const char *__restrict src,
                                char **__restrict str_end, int base) {
  unsigned long long result = 0;

  if (base < 0 || base == 1 || base > 36) {
    errno = EINVAL; // NOLINT
    return 0;
  }

  src = first_non_whitespace(src);

  char result_sign = '+';
  if (*src == '+' || *src == '-') {
    result_sign = *src;
    ++src;
  }

  if (base == 0) {
    base = infer_base(&src);
  } else if (base == 16 && *src == '0' && (*(src + 1) | 32) == 'x') {
    src = src + 2;
  }

  unsigned long long const ABS_MAX =
      (result_sign == '+' ? LLONG_MAX
                          : static_cast<unsigned long long>(LLONG_MAX) + 1);
  unsigned long long const ABS_MAX_DIV_BY_BASE = ABS_MAX / base;
  while (isalnum(*src)) {
    int cur_digit = b36_char_to_int(*src);
    if (cur_digit >= base)
      break;
    if (result > ABS_MAX_DIV_BY_BASE) {
      result = ABS_MAX;
      errno = ERANGE; // NOLINT
      break;
    }
    result = result * base;
    if (result > ABS_MAX - cur_digit) {
      result = ABS_MAX;
      errno = ERANGE; // NOLINT
      break;
    }
    result = result + cur_digit;

    ++src;
  }

  if (str_end != nullptr)
    *str_end = const_cast<char *>(src);
  if (result_sign == '+')
    return result;
  else
    return -result;
}

} // namespace internal
} // namespace __llvm_libc

#endif // LIBC_SRC_STDLIB_STDLIB_UTILS_H

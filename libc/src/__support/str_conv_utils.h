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
#include "utils/CPP/Limits.h"
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

// checks if the next 3 characters of the string pointer are the start of a
// hexadecimal number. Does not advance the string pointer.
static inline bool is_hex_start(const char *__restrict src) {
  return *src == '0' && (*(src + 1) | 32) == 'x' && isalnum(*(src + 2)) &&
         b36_char_to_int(*(src + 2)) < 16;
}

// Takes the address of the string pointer and parses the base from the start of
// it. This will advance the string pointer.
static inline int infer_base(const char *__restrict *__restrict src) {
  if (is_hex_start(*src)) {
    (*src) += 2;
    return 16;
  } else if (**src == '0') {
    ++(*src);
    return 8;
  } else {
    return 10;
  }
}

// Takes a pointer to a string, a pointer to a string pointer, and the base to
// convert to. This function is used as the backend for all of the string to int
// functions.
template <class T>
static inline T strtointeger(const char *__restrict src,
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
  } else if (base == 16 && is_hex_start(src)) {
    src = src + 2;
  }

  constexpr bool is_unsigned = (__llvm_libc::cpp::NumericLimits<T>::min() == 0);
  const bool is_positive = (result_sign == '+');
  unsigned long long constexpr NEGATIVE_MAX =
      !is_unsigned ? static_cast<unsigned long long>(
                         __llvm_libc::cpp::NumericLimits<T>::max()) +
                         1
                   : __llvm_libc::cpp::NumericLimits<T>::max();
  unsigned long long const ABS_MAX =
      (is_positive ? __llvm_libc::cpp::NumericLimits<T>::max() : NEGATIVE_MAX);
  unsigned long long const ABS_MAX_DIV_BY_BASE = ABS_MAX / base;
  while (isalnum(*src)) {
    int cur_digit = b36_char_to_int(*src);
    if (cur_digit >= base)
      break;

    ++src;

    // If the number has already hit the maximum value for the current type then
    // the result cannot change, but we still need to advance src to the end of
    // the number.
    if (result == ABS_MAX) {
      errno = ERANGE; // NOLINT
      continue;
    }

    if (result > ABS_MAX_DIV_BY_BASE) {
      result = ABS_MAX;
      errno = ERANGE; // NOLINT
    } else {
      result = result * base;
    }
    if (result > ABS_MAX - cur_digit) {
      result = ABS_MAX;
      errno = ERANGE; // NOLINT
    } else {
      result = result + cur_digit;
    }
  }

  if (str_end != nullptr)
    *str_end = const_cast<char *>(src);

  if (result == ABS_MAX) {
    if (is_positive || is_unsigned)
      return __llvm_libc::cpp::NumericLimits<T>::max();
    else // T is signed and there is a negative overflow
      return __llvm_libc::cpp::NumericLimits<T>::min();
  }

  return is_positive ? static_cast<T>(result) : -static_cast<T>(result);
}

} // namespace internal
} // namespace __llvm_libc

#endif // LIBC_SRC_STDLIB_STDLIB_UTILS_H

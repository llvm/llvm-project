//===-- String to integer conversion utils ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_SRC_SUPPORT_STR_TO_INTEGER_H
#define LIBC_SRC_SUPPORT_STR_TO_INTEGER_H

#include "src/__support/CPP/limits.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/str_to_num_result.h"
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
// it. This function will advance |src| to the first valid digit in the inferred
// base.
static inline int infer_base(const char *__restrict *__restrict src) {
  // A hexadecimal number is defined as "the prefix 0x or 0X followed by a
  // sequence of the deimal digits and the letters a (or A) through f (or F)
  // with values 10 through 15 respectively." (C standard 6.4.4.1)
  if (is_hex_start(*src)) {
    (*src) += 2;
    return 16;
  } // An octal number is defined as "the prefix 0 optionally followed by a
    // sequence of the digits 0 through 7 only" (C standard 6.4.4.1) and so any
    // number that starts with 0, including just 0, is an octal number.
  else if (**src == '0') {
    return 8;
  } // A decimal number is defined as beginning "with a nonzero digit and
    // consist[ing] of a sequence of decimal digits." (C standard 6.4.4.1)
  else {
    return 10;
  }
}

// Takes a pointer to a string and the base to convert to. This function is used
// as the backend for all of the string to int functions.
template <class T>
static inline StrToNumResult<T> strtointeger(const char *__restrict src,
                                             int base) {
  unsigned long long result = 0;
  bool is_number = false;
  const char *original_src = src;
  int error_val = 0;

  if (base < 0 || base == 1 || base > 36) {
    error_val = EINVAL;
    return {0, 0, error_val};
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

  constexpr bool IS_UNSIGNED = (cpp::numeric_limits<T>::min() == 0);
  const bool is_positive = (result_sign == '+');
  unsigned long long constexpr NEGATIVE_MAX =
      !IS_UNSIGNED
          ? static_cast<unsigned long long>(cpp::numeric_limits<T>::max()) + 1
          : cpp::numeric_limits<T>::max();
  unsigned long long const abs_max =
      (is_positive ? cpp::numeric_limits<T>::max() : NEGATIVE_MAX);
  unsigned long long const abs_max_div_by_base = abs_max / base;
  while (isalnum(*src)) {
    int cur_digit = b36_char_to_int(*src);
    if (cur_digit >= base)
      break;

    is_number = true;
    ++src;

    // If the number has already hit the maximum value for the current type then
    // the result cannot change, but we still need to advance src to the end of
    // the number.
    if (result == abs_max) {
      error_val = ERANGE;
      continue;
    }

    if (result > abs_max_div_by_base) {
      result = abs_max;
      error_val = ERANGE;
    } else {
      result = result * base;
    }
    if (result > abs_max - cur_digit) {
      result = abs_max;
      error_val = ERANGE;
    } else {
      result = result + cur_digit;
    }
  }

  ptrdiff_t str_len = is_number ? (src - original_src) : 0;

  if (result == abs_max) {
    if (is_positive || IS_UNSIGNED)
      return {cpp::numeric_limits<T>::max(), str_len, error_val};
    else // T is signed and there is a negative overflow
      return {cpp::numeric_limits<T>::min(), str_len, error_val};
  }

  return {is_positive ? static_cast<T>(result) : -static_cast<T>(result),
          str_len, error_val};
}

} // namespace internal
} // namespace __llvm_libc

#endif // LIBC_SRC_SUPPORT_STR_TO_INTEGER_H

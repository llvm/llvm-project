//===-- Widechar string to integer conversion utils -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_WCS_TO_INTEGER_H
#define LLVM_LIBC_SRC___SUPPORT_WCS_TO_INTEGER_H

#include "hdr/errno_macros.h" // For ERANGE
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/CPP/type_traits/make_unsigned.h"
#include "src/__support/big_int.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/str_to_num_result.h"
#include "src/__support/uint128.h"
#include "src/__support/wctype_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// Returns the idx of the first character in src that is not a whitespace
// character (as determined by iswspace())
LIBC_INLINE size_t
first_non_whitespace(const wchar_t *__restrict src,
                     size_t src_len = cpp::numeric_limits<size_t>::max()) {
  size_t src_cur = 0;
  while (src_cur < src_len && internal::iswspace(src[src_cur])) {
    ++src_cur;
  }
  return src_cur;
}

// checks if the next 3 characters of the string pointer are the start of a
// hexadecimal number. Does not advance the string pointer.
LIBC_INLINE bool
is_hex_start(const wchar_t *__restrict src,
             size_t src_len = cpp::numeric_limits<size_t>::max()) {
  if (src_len < 3)
    return false;
  return *src == L'0' && towlower(*(src + 1)) == L'x' && iswalnum(*(src + 2)) &&
         b36_wchar_to_int(*(src + 2)) < 16;
}

// Takes the address of the string pointer and parses the base from the start of
// it.
LIBC_INLINE int infer_base(const wchar_t *__restrict src, size_t src_len) {
  // A hexadecimal number is defined as "the prefix 0x or 0X followed by a
  // sequence of the decimal digits and the letters a (or A) through f (or F)
  // with values 10 through 15 respectively." (C standard 6.4.4.1)
  if (is_hex_start(src, src_len))
    return 16;
  // An octal number is defined as "the prefix 0 optionally followed by a
  // sequence of the digits 0 through 7 only" (C standard 6.4.4.1) and so any
  // number that starts with 0, including just 0, is an octal number.
  if (src_len > 0 && src[0] == L'0')
    return 8;
  // A decimal number is defined as beginning "with a nonzero digit and
  // consist[ing] of a sequence of decimal digits." (C standard 6.4.4.1)
  return 10;
}

template <class T>
LIBC_INLINE StrToNumResult<T>
wcstointeger(const wchar_t *__restrict src, int base,
             const size_t src_len = cpp::numeric_limits<size_t>::max()) {
  using ResultType = make_integral_or_big_int_unsigned_t<T>;

  ResultType result = 0;

  bool is_number = false;
  size_t src_cur = 0;
  int error_val = 0;

  if (src_len == 0)
    return {0, 0, 0};

  if (base < 0 || base == 1 || base > 36)
    return {0, 0, EINVAL};

  src_cur = first_non_whitespace(src, src_len);

  wchar_t result_sign = L'+';
  if (src[src_cur] == L'+' || src[src_cur] == L'-') {
    result_sign = src[src_cur];
    ++src_cur;
  }

  if (base == 0)
    base = infer_base(src + src_cur, src_len - src_cur);

  if (base == 16 && is_hex_start(src + src_cur, src_len - src_cur))
    src_cur = src_cur + 2;

  constexpr bool IS_UNSIGNED = cpp::is_unsigned_v<T>;
  const bool is_positive = (result_sign == L'+');

  ResultType constexpr NEGATIVE_MAX =
      !IS_UNSIGNED ? static_cast<ResultType>(cpp::numeric_limits<T>::max()) + 1
                   : cpp::numeric_limits<T>::max();
  ResultType const abs_max =
      (is_positive ? cpp::numeric_limits<T>::max() : NEGATIVE_MAX);
  ResultType const abs_max_div_by_base =
      abs_max / static_cast<ResultType>(base);

  while (src_cur < src_len && iswalnum(src[src_cur])) {
    int cur_digit = b36_wchar_to_int(src[src_cur]);
    if (cur_digit >= base)
      break;

    is_number = true;
    ++src_cur;

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
      result = result * static_cast<ResultType>(base);
    }
    if (result > abs_max - static_cast<ResultType>(cur_digit)) {
      result = abs_max;
      error_val = ERANGE;
    } else {
      result = result + static_cast<ResultType>(cur_digit);
    }
  }

  ptrdiff_t str_len = is_number ? static_cast<ptrdiff_t>(src_cur) : 0;

  if (error_val == ERANGE) {
    if (is_positive || IS_UNSIGNED)
      return {cpp::numeric_limits<T>::max(), str_len, error_val};
    else // T is signed and there is a negative overflow
      return {cpp::numeric_limits<T>::min(), str_len, error_val};
  }

  return {static_cast<T>(is_positive ? result : -result), str_len, error_val};
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_WCS_TO_INTEGER_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SRC_INCLUDE_FROM_CHARS_FLOATING_POINT_H
#define _LIBCPP_SRC_INCLUDE_FROM_CHARS_FLOATING_POINT_H

// NEVER DO THIS FOR REAL, this is just for demonstration purposes.
#define LIBC_NAMESPACE libc_namespace_in_libcxx

// This header is in the shared LLVM-libc header library.
#include "shared/str_to_float.h"

#include <__assert>
#include <__config>
#include <charconv>
#include <limits>
#include <type_traits>

// Included for the _Floating_type_traits class
#include "to_chars_floating_point.h"

_LIBCPP_BEGIN_NAMESPACE_STD

template <typename _Tp, __enable_if_t<std::is_floating_point<_Tp>::value, int> = 0>
from_chars_result from_chars_floating_point(const char* __first, const char* __last, _Tp& __value, chars_format __fmt) {
  using _Traits    = _Floating_type_traits<_Tp>;
  using _Uint_type = typename _Traits::_Uint_type;
  ptrdiff_t length = __last - __first;
  _LIBCPP_ASSERT_INTERNAL(length > 0, "");

  // hacky parsing code as example. Not intended for actual use. I'm just going to handle the base 10
  // chars_format::general case. Also, no sign, inf, or nan handling.
  _LIBCPP_ASSERT_INTERNAL(__fmt == std::chars_format::general, "");

  const char* src = __first; // rename to match the libc code copied for this section.

  _Uint_type mantissa            = 0;
  int exponent                   = 0;
  bool truncated                 = false;
  bool seen_digit                = false;
  bool after_decimal             = false;
  size_t index                   = 0;
  const size_t BASE              = 10;
  constexpr char EXPONENT_MARKER = 'e';
  constexpr char DECIMAL_POINT   = '.';

  // The loop fills the mantissa with as many digits as it can hold
  const _Uint_type bitstype_max_div_by_base = numeric_limits<_Uint_type>::max() / BASE;
  while (index < static_cast<size_t>(length)) {
    if (LIBC_NAMESPACE::internal::isdigit(src[index])) {
      uint32_t digit = src[index] - '0';
      seen_digit     = true;

      if (mantissa < bitstype_max_div_by_base) {
        mantissa = (mantissa * BASE) + digit;
        if (after_decimal) {
          --exponent;
        }
      } else {
        if (digit > 0)
          truncated = true;
        if (!after_decimal)
          ++exponent;
      }

      ++index;
      continue;
    }
    if (src[index] == DECIMAL_POINT) {
      if (after_decimal) {
        break; // this means that src[index] points to a second decimal point, ending the number.
      }
      after_decimal = true;
      ++index;
      continue;
    }
    // The character is neither a digit nor a decimal point.
    break;
  }

  if (!seen_digit)
    return {src + index, {}};

  if (index < static_cast<size_t>(length) && LIBC_NAMESPACE::internal::tolower(src[index]) == EXPONENT_MARKER) {
    bool has_sign = false;
    if (index + 1 < static_cast<size_t>(length) && (src[index + 1] == '+' || src[index + 1] == '-')) {
      has_sign = true;
    }
    if (index + 1 + static_cast<size_t>(has_sign) < static_cast<size_t>(length) &&
        LIBC_NAMESPACE::internal::isdigit(src[index + 1 + static_cast<size_t>(has_sign)])) {
      ++index;
      auto result = LIBC_NAMESPACE::internal::strtointeger<int32_t>(src + index, 10);
      // if (result.has_error())
      //   output.error = result.error;
      int32_t add_to_exponent = result.value;
      index += result.parsed_len;

      // Here we do this operation as int64 to avoid overflow.
      int64_t temp_exponent = static_cast<int64_t>(exponent) + static_cast<int64_t>(add_to_exponent);

      // If the result is in the valid range, then we use it. The valid range is
      // also within the int32 range, so this prevents overflow issues.
      if (temp_exponent > LIBC_NAMESPACE::fputil::FPBits<_Tp>::MAX_BIASED_EXPONENT) {
        exponent = LIBC_NAMESPACE::fputil::FPBits<_Tp>::MAX_BIASED_EXPONENT;
      } else if (temp_exponent < -LIBC_NAMESPACE::fputil::FPBits<_Tp>::MAX_BIASED_EXPONENT) {
        exponent = -LIBC_NAMESPACE::fputil::FPBits<_Tp>::MAX_BIASED_EXPONENT;
      } else {
        exponent = static_cast<int32_t>(temp_exponent);
      }
    }
  }

  LIBC_NAMESPACE::internal::ExpandedFloat<_Tp> expanded_float = {0, 0};
  if (mantissa != 0) {
    auto temp = LIBC_NAMESPACE::shared::decimal_exp_to_float<_Tp>(
        {mantissa, exponent}, truncated, LIBC_NAMESPACE::internal::RoundDirection::Nearest, src, length);
    expanded_float = temp.num;
    // Note: there's also an error value in temp.error. I'm not doing that error handling right now though.
  }

  auto result = LIBC_NAMESPACE::fputil::FPBits<_Tp>();
  result.set_mantissa(expanded_float.mantissa);
  result.set_biased_exponent(expanded_float.exponent);
  __value = result.get_val();
  return {src + index, {}};
}

_LIBCPP_END_NAMESPACE_STD

#endif //_LIBCPP_SRC_INCLUDE_FROM_CHARS_FLOATING_POINT_H

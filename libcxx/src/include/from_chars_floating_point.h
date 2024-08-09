//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SRC_INCLUDE_FROM_CHARS_FLOATING_POINT_H
#define _LIBCPP_SRC_INCLUDE_FROM_CHARS_FLOATING_POINT_H

// This header is in the shared LLVM-libc header library.
#include "shared/str_to_float.h"

#include <__assert>
#include <__config>
#include <cctype>
#include <charconv>
#include <concepts>
#include <limits>
#include <type_traits>

// Included for the _Floating_type_traits class
#include "to_chars_floating_point.h"

_LIBCPP_BEGIN_NAMESPACE_STD

// Parses an infinity string.
// Valid strings are case insentitive and contain INF or INFINITY.
//
// - __first is the first argument to std::from_chars. When the string is invalid
//   this value is returned as ptr in the result.
// - __last is the last argument of std::from_chars.
// - __value is the value argument of std::from_chars,
// - __ptr is the current position is the input string. This is points beyond
//   the initial I character.
// - __negative whether a valid string represents -inf or +inf.
template <floating_point _Fp>
from_chars_result __from_chars_floating_point_inf(
    const char* const __first, const char* __last, _Fp& __value, const char* __ptr, bool __negative) {
  if (__last - __ptr < 2) [[unlikely]]
    return {__first, errc::invalid_argument};

  if (std::tolower(__ptr[0]) != 'n' || std::tolower(__ptr[1]) != 'f') [[unlikely]]
    return {__first, errc::invalid_argument};

  __ptr += 2;

  // At this point the result is valid and contains INF.
  // When the remaining part contains INITY this will be consumed. Otherwise
  // only INF is consumed. For example INFINITZ will consume INF and ignore
  // INITZ.

  if (__last - __ptr >= 5              //
      && std::tolower(__ptr[0]) == 'i' //
      && std::tolower(__ptr[1]) == 'n' //
      && std::tolower(__ptr[2]) == 'i' //
      && std::tolower(__ptr[3]) == 't' //
      && std::tolower(__ptr[4]) == 'y')
    __ptr += 5;

  if constexpr (numeric_limits<_Fp>::has_infinity) {
    if (__negative)
      __value = -std::numeric_limits<_Fp>::infinity();
    else
      __value = std::numeric_limits<_Fp>::infinity();

    return {__ptr, std::errc{}};
  } else {
    return {__ptr, errc::result_out_of_range};
  }
}

// Parses a nan string.
// Valid strings are case insentitive and contain INF or INFINITY.
//
// - __first is the first argument to std::from_chars. When the string is invalid
//   this value is returned as ptr in the result.
// - __last is the last argument of std::from_chars.
// - __value is the value argument of std::from_chars,
// - __ptr is the current position is the input string. This is points beyond
//   the initial N character.
// - __negative whether a valid string represents -nan or +nan.
template <floating_point _Fp>
from_chars_result __from_chars_floating_point_nan(
    const char* const __first, const char* __last, _Fp& __value, const char* __ptr, bool __negative) {
  if (__last - __ptr < 2) [[unlikely]]
    return {__first, errc::invalid_argument};

  if (std::tolower(__ptr[0]) != 'a' || std::tolower(__ptr[1]) != 'n') [[unlikely]]
    return {__first, errc::invalid_argument};

  __ptr += 2;

  // At this point the result is valid and contains NAN. When the remaining
  // part contains ( n-char-sequence_opt ) this will be consumed. Otherwise
  // only NAN is consumed. For example NAN(abcd will consume NAN and ignore
  // (abcd.
  if (__last - __ptr >= 2 && __ptr[0] == '(') {
    size_t __offset = 1;
    do {
      if (__ptr[__offset] == ')') {
        __ptr += __offset + 1;
        break;
      }
      if (__ptr[__offset] != '_' && !std::isalnum(__ptr[__offset]))
        break;
      ++__offset;
    } while (__ptr + __offset != __last);
  }

  if (__negative)
    __value = -std::numeric_limits<_Fp>::quiet_NaN();
  else
    __value = std::numeric_limits<_Fp>::quiet_NaN();

  return {__ptr, std::errc{}};
}

template <floating_point _Fp>
from_chars_result __from_chars_floating_point_decimal(
    const char* const __first,
    const char* __last,
    _Fp& __value,
    chars_format __fmt,
    const char* __ptr,
    bool __negative) {
  using _Traits    = _Floating_type_traits<_Fp>;
  using _Uint_type = typename _Traits::_Uint_type;

  const char* src  = __ptr; // rename to match the libc code copied for this section.
  ptrdiff_t length = __last - src;
  _LIBCPP_ASSERT_INTERNAL(length > 0, "");

  _Uint_type mantissa            = 0;
  int exponent                   = 0;
  bool truncated                 = false;
  bool seen_digit                = false;
  bool has_valid_exponent        = false;
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
    return {__first, errc::invalid_argument};

  if (index < static_cast<size_t>(length) && LIBC_NAMESPACE::internal::tolower(src[index]) == EXPONENT_MARKER) {
    bool has_sign = false;
    if (index + 1 < static_cast<size_t>(length) && (src[index + 1] == '+' || src[index + 1] == '-')) {
      has_sign = true;
    }
    if (index + 1 + static_cast<size_t>(has_sign) < static_cast<size_t>(length) &&
        LIBC_NAMESPACE::internal::isdigit(src[index + 1 + static_cast<size_t>(has_sign)])) {
      has_valid_exponent = true;
      ++index;
      auto result =
          LIBC_NAMESPACE::internal::strtointeger<int32_t>(src + index, 10, static_cast<size_t>(length) - index);
      // if (result.has_error())
      //   output.error = result.error;
      int32_t add_to_exponent = result.value;
      index += result.parsed_len;

      // Here we do this operation as int64 to avoid overflow.
      int64_t temp_exponent = static_cast<int64_t>(exponent) + static_cast<int64_t>(add_to_exponent);

      // If the result is in the valid range, then we use it. The valid range is
      // also within the int32 range, so this prevents overflow issues.
      if (temp_exponent > LIBC_NAMESPACE::fputil::FPBits<_Fp>::MAX_BIASED_EXPONENT) {
        exponent = LIBC_NAMESPACE::fputil::FPBits<_Fp>::MAX_BIASED_EXPONENT;
      } else if (temp_exponent < -LIBC_NAMESPACE::fputil::FPBits<_Fp>::MAX_BIASED_EXPONENT) {
        exponent = -LIBC_NAMESPACE::fputil::FPBits<_Fp>::MAX_BIASED_EXPONENT;
      } else {
        exponent = static_cast<int32_t>(temp_exponent);
      }
    }
  }

  // [charconv.from.chars]
  switch (__fmt) {
  case chars_format::scientific:
    // 6.2 if fmt has chars_format::scientific set but not chars_format::fixed,
    // the otherwise optional exponent part shall appear;
    if (!has_valid_exponent)
      return {__first, errc::invalid_argument};
    break;
  case chars_format::fixed:
    // 6.3 if fmt has chars_format::fixed set but not chars_format::scientific,
    // the optional exponent part shall not appear;
    if (has_valid_exponent)
      return {__first, errc::invalid_argument};
    break;
  case chars_format::general:
  case chars_format::hex: // impossible but it silences the compiler
    break;
  }

  LIBC_NAMESPACE::internal::ExpandedFloat<_Fp> expanded_float = {0, 0};
  errc status{};
  if (mantissa != 0) {
    auto temp = LIBC_NAMESPACE::shared::decimal_exp_to_float<_Fp>(
        {mantissa, exponent}, truncated, LIBC_NAMESPACE::internal::RoundDirection::Nearest, src, length);
    expanded_float = temp.num;
    if (temp.error == ERANGE) {
      status = errc::result_out_of_range;
    }
  }

  auto result = LIBC_NAMESPACE::fputil::FPBits<_Fp>();
  result.set_mantissa(expanded_float.mantissa);
  result.set_biased_exponent(expanded_float.exponent);

  // C17 7.12.1/6
  // The result underflows if the magnitude of the mathematical result is so
  // small that the mathematical re- sult cannot be represented, without
  // extraordinary roundoff error, in an object of the specified type.237) If
  // the result underflows, the function returns an implementation-defined
  // value whose magnitude is no greater than the smallest normalized positive
  // number in the specified type; if the integer expression math_errhandling
  // & MATH_ERRNO is nonzero, whether errno acquires the value ERANGE is
  // implementation-defined; if the integer expression math_errhandling &
  // MATH_ERREXCEPT is nonzero, whether the "underflow" floating-point
  // exception is raised is implementation-defined.
  //
  // LLLVM-LIBC sets ERAGNE for subnormal values
  //
  // [charconv.from.chars]/1
  //   ... If the parsed value is not in the range representable by the type of
  //   value, value is unmodified and the member ec of the return value is
  //   equal to errc::result_out_of_range. ...
  //
  // Undo the ERANGE for subnormal values.
  if (status == errc::result_out_of_range && result.is_subnormal() && !result.is_zero())
    status = errc{};

  if (__negative)
    __value = -result.get_val();
  else
    __value = result.get_val();

  return {src + index, status};
}

template <floating_point _Fp>
from_chars_result
__from_chars_floating_point(const char* const __first, const char* __last, _Fp& __value, chars_format __fmt) {
  if (__first == __last) [[unlikely]]
    return {__first, errc::invalid_argument};

  const char* __ptr = __first;
  bool __negative   = *__ptr == '-';
  if (__negative) {
    ++__ptr;
    if (__ptr == __last) [[unlikely]]
      return {__first, errc::invalid_argument};
  }

  // [charconv.from.chars]
  //   [Note 1: If the pattern allows for an optional sign, but the string has
  //   no digit characters following the sign, no characters match the pattern.
  //   — end note]
  // This is true for integrals, floating point allows -.0
  switch (std::tolower(*__ptr)) {
  case 'i':
    // TODO Evaluate the other implementations
    // [charconv.from.chars]/6.2
    //   if fmt has chars_format::scientific set but not chars_format::fixed,
    //   the otherwise optional exponent part shall appear;
    // Since INF/NAN do not have an exponent this value is not valid.
    // See LWG3456
    if (__fmt == chars_format::scientific)
      return {__first, errc::invalid_argument};

    return __from_chars_floating_point_inf(__first, __last, __value, __ptr + 1, __negative);
  case 'n':
    // TODO Evaluate the other implementations
    // [charconv.from.chars]/6.2
    //   if fmt has chars_format::scientific set but not chars_format::fixed,
    //   the otherwise optional exponent part shall appear;
    // Since INF/NAN do not have an exponent this value is not valid.
    // See LWG3456
    if (__fmt == chars_format::scientific)
      return {__first, errc::invalid_argument};
    if constexpr (numeric_limits<_Fp>::has_quiet_NaN)
      return __from_chars_floating_point_nan(__first, __last, __value, __ptr + 1, __negative);
    return {__first, errc::invalid_argument};
  }

#if 1
  _LIBCPP_ASSERT_INTERNAL(__fmt != std::chars_format::hex, "");
#else
  if (__fmt == chars_format::hex)
    return std::__from_chars_floating_point_hex(__first, __last, __value);
#endif

  return std::__from_chars_floating_point_decimal(__first, __last, __value, __fmt, __ptr, __negative);
}

_LIBCPP_END_NAMESPACE_STD

#endif //_LIBCPP_SRC_INCLUDE_FROM_CHARS_FLOATING_POINT_H

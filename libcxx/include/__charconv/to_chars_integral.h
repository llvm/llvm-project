// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHARCONV_TO_CHARS_INTEGRAL_H
#define _LIBCPP___CHARCONV_TO_CHARS_INTEGRAL_H

#include <__algorithm/copy_n.h>
#include <__algorithm/simd_utils.h>
#include <__assert>
#include <__bit/countl.h>
#include <__charconv/tables.h>
#include <__charconv/to_chars_base_10.h>
#include <__charconv/to_chars_result.h>
#include <__charconv/traits.h>
#include <__config>
#include <__cstddef/ptrdiff_t.h>
#include <__system_error/errc.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_integral.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_signed.h>
#include <__type_traits/make_32_64_or_128_bit.h>
#include <__type_traits/make_unsigned.h>
#include <__utility/unreachable.h>
#include <array>
#include <cstdint>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <typename _Tp>
inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI __to_chars_result
__to_chars_itoa(char* __first, char* __last, _Tp __value, false_type);

template <typename _Tp>
inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI __to_chars_result
__to_chars_itoa(char* __first, char* __last, _Tp __value, true_type) {
  auto __x = std::__to_unsigned_like(__value);
  if (__value < 0 && __first != __last) {
    *__first++ = '-';
    __x        = std::__complement(__x);
  }

  return std::__to_chars_itoa(__first, __last, __x, false_type());
}

template <typename _Tp>
inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI __to_chars_result
__to_chars_itoa(char* __first, char* __last, _Tp __value, false_type) {
  using __tx  = __itoa::__traits<_Tp>;
  auto __diff = __last - __first;

  if (__tx::digits <= __diff || __tx::__width(__value) <= __diff)
    return {__tx::__convert(__first, __value), errc(0)};
  else
    return {__last, errc::value_too_large};
}

#  if _LIBCPP_HAS_INT128
template <>
inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI __to_chars_result
__to_chars_itoa(char* __first, char* __last, __uint128_t __value, false_type) {
  // When the value fits in 64-bits use the 64-bit code path. This reduces
  // the number of expensive calculations on 128-bit values.
  //
  // NOTE the 128-bit code path requires this optimization.
  if (__value <= numeric_limits<uint64_t>::max())
    return __to_chars_itoa(__first, __last, static_cast<uint64_t>(__value), false_type());

  using __tx  = __itoa::__traits<__uint128_t>;
  auto __diff = __last - __first;

  if (__tx::digits <= __diff || __tx::__width(__value) <= __diff)
    return {__tx::__convert(__first, __value), errc(0)};
  else
    return {__last, errc::value_too_large};
}
#  endif

template <class _Tp, __enable_if_t<!is_signed<_Tp>::value, int> = 0>
inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI __to_chars_result
__to_chars_integral(char* __first, char* __last, _Tp __value, int __base);

template <class _Tp, __enable_if_t<is_signed<_Tp>::value, int> = 0>
inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI __to_chars_result
__to_chars_integral(char* __first, char* __last, _Tp __value, int __base) {
  auto __x = std::__to_unsigned_like(__value);
  if (__value < 0 && __first != __last) {
    *__first++ = '-';
    __x        = std::__complement(__x);
  }

  return std::__to_chars_integral(__first, __last, __x, __base);
}

namespace __itoa {

template <size_t _Duplicate, size_t _IndexCount>
constexpr array<size_t, _IndexCount * _Duplicate> __build_indices() {
  array<size_t, _IndexCount * _Duplicate> __ret;
  for (size_t __i = 0; __i != _IndexCount; ++__i) {
    for (size_t __k = 0; __k != _Duplicate; ++__k)
      __ret[__i * _Duplicate + __k] = _IndexCount - __i - 1;
  }
  return __ret;
}

_LIBCPP_DIAGNOSTIC_PUSH
// TODO: remove this once all supported compilers diagnose functions correctly
_LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wpsabi")
// This is marked `always_inline` because it interacts with simd vectors
template <size_t _Duplicate, size_t _VecSize>
[[__gnu__::__always_inline__]] auto __duplicate_vector_entries(__simd_vector<char, _VecSize> __vals) {
  static constexpr auto __indices = __itoa::__build_indices<_Duplicate, _VecSize>();
  return [&]<size_t... _Indices> [[__gnu__::__always_inline__]] (index_sequence<_Indices...>) {
    return __builtin_shufflevector(__vals, __vals, __indices[_Indices]...);
  }(make_index_sequence<_Duplicate * _VecSize>());
}
_LIBCPP_DIAGNOSTIC_POP

template <unsigned _Base>
struct _LIBCPP_HIDDEN __integral;

template <>
struct _LIBCPP_HIDDEN __integral<2> {
  template <typename _Tp>
  _LIBCPP_HIDE_FROM_ABI static _LIBCPP_CONSTEXPR int __width(_Tp __value) _NOEXCEPT {
    // If value == 0 still need one digit. If the value != this has no
    // effect since the code scans for the most significant bit set.
    return numeric_limits<_Tp>::digits - std::__countl_zero(__value | 1);
  }

  template <typename _Tp>
  _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI static __to_chars_result
  __to_chars(char* __first, char* __last, _Tp __value) {
#if _LIBCPP_VECTORIZE_ALGORITHMS && __has_builtin(__builtin_masked_store)
    if (!__libcpp_is_constant_evaluated() && __last - __first >= numeric_limits<_Tp>::digits) {
      // Move the to-be-converted bits into the high bits, so that they are printed at the start.
      auto __char_count = __width(__value);
      auto __shift      = std::__countl_zero(__value | 1);
      __value <<= __shift;

      // Move the value into a vector and chop it up into its constituent bytes
      auto __chopped    = __simd_vector<char, sizeof(_Tp)>(__simd_vector<_Tp, 1>(__value));

      // Duplicate values so we can extract the appropriate bits in multiple positions
      auto __characters = __itoa::__duplicate_vector_entries<8>(__chopped);

      // This is marked `always_inline` because it interacts with simd vectors
      _LIBCPP_DIAGNOSTIC_PUSH
      // TODO: remove this once all supported compilers diagnose functions correctly
      _LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wpsabi")
      auto __shifts = []<size_t... _Indices> [[__gnu__::__always_inline__]] (index_sequence<_Indices...>) {
        return __simd_vector<char, 8 * sizeof(_Tp)>{(7 - _Indices % 8)...};
      }(make_index_sequence<8 * sizeof(_Tp)>());
      _LIBCPP_DIAGNOSTIC_POP

      // Check the appropriate bit for the position and set the character to 0 or 1 depending on whether it's set.
      __characters = (__characters & __shifts) == 0 ? '0' : '1';

      // Store all the characters, no matter whether they're part of the value or not. This is only safe if the buffer
      // we've been given is large enough.
      // TODO: Generate a mask on platforms which have native masked stores and use this code path unconditionally.
      __builtin_masked_store(__simd_vector<bool, 8 * sizeof(_Tp)>(true), __characters, __first);
      return {__first + __char_count, errc(0)};
    }
#endif

    ptrdiff_t __cap = __last - __first;
    int __n         = __width(__value);
    if (__n > __cap)
      return {__last, errc::value_too_large};

    __last                   = __first + __n;
    char* __p                = __last;
    const unsigned __divisor = 16;
    while (__value > __divisor) {
      unsigned __c = __value % __divisor;
      __value /= __divisor;
      __p -= 4;
      std::copy_n(&__base_2_lut[4 * __c], 4, __p);
    }
    do {
      unsigned __c = __value % 2;
      __value /= 2;
      *--__p = "01"[__c];
    } while (__value != 0);
    return {__last, errc(0)};
  }
};

template <>
struct _LIBCPP_HIDDEN __integral<8> {
  template <typename _Tp>
  _LIBCPP_HIDE_FROM_ABI static _LIBCPP_CONSTEXPR int __width(_Tp __value) _NOEXCEPT {
    // If value == 0 still need one digit. If the value != this has no
    // effect since the code scans for the most significat bit set.
    return ((numeric_limits<_Tp>::digits - std::__countl_zero(__value | 1)) + 2) / 3;
  }

  template <typename _Tp>
  _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI static __to_chars_result
  __to_chars(char* __first, char* __last, _Tp __value) {
    ptrdiff_t __cap = __last - __first;
    int __n         = __width(__value);
    if (__n > __cap)
      return {__last, errc::value_too_large};

    __last             = __first + __n;
    char* __p          = __last;
    unsigned __divisor = 64;
    while (__value > __divisor) {
      unsigned __c = __value % __divisor;
      __value /= __divisor;
      __p -= 2;
      std::copy_n(&__base_8_lut[2 * __c], 2, __p);
    }
    do {
      unsigned __c = __value % 8;
      __value /= 8;
      *--__p = "01234567"[__c];
    } while (__value != 0);
    return {__last, errc(0)};
  }
};

template <>
struct _LIBCPP_HIDDEN __integral<16> {
  template <typename _Tp>
  _LIBCPP_HIDE_FROM_ABI static _LIBCPP_CONSTEXPR int __width(_Tp __value) _NOEXCEPT {
    // If value == 0 still need one digit. If the value != this has no
    // effect since the code scans for the most significat bit set.
    return (numeric_limits<_Tp>::digits - std::__countl_zero(__value | 1) + 3) / 4;
  }

  template <typename _Tp>
  _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI static __to_chars_result
  __to_chars(char* __first, char* __last, _Tp __value) {
#if _LIBCPP_VECTORIZE_ALGORITHMS && __has_builtin(__builtin_masked_store)
    if (!__libcpp_is_constant_evaluated() && __last - __first >= (numeric_limits<_Tp>::digits / 4)) {
      // Lambdas are marked as `always_inline` because they return vectors, which aren't ABI-stable

      // Move the to-be-converted bits into the high bits, so that they are printed at the start. Note that the shift is
      // 4-bit aligned, so that the value of the characters doesn't change.
      auto __char_count = __width(__value);
      auto __shift      = std::__countl_zero(__value | 1) & ~3;
      __value <<= __shift;

      // Move the value into a vector and chop it up into its constituent bytes
      auto __chopped    = __simd_vector<char, sizeof(_Tp)>(__simd_vector<_Tp, 1>(__value));

      // Duplicate values so we can extract the appropriate bits in multiple positions
      auto __characters = __itoa::__duplicate_vector_entries<2>(__chopped);

      // This is marked `always_inline` because it interacts with simd vectors
      _LIBCPP_DIAGNOSTIC_PUSH
      // TODO: remove this once all supported compilers diagnose functions correctly
      _LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wpsabi")
      auto __shifts     = []<size_t... _Indices> [[__gnu__::__always_inline__]] (index_sequence<_Indices...>) {
        return __simd_vector<char, 2 * sizeof(_Tp)>{(_Indices % 2 == 0 ? 4 : 0)...};
      }(make_index_sequence<2 * sizeof(_Tp)>());
      _LIBCPP_DIAGNOSTIC_POP

      // Extract the bits we want for a given offset
      __characters = (__characters >> __shifts) & 15;

      // Convert the value into the hexadecimal character
      __characters = (__characters >= 10 ? char('a' - 10) : '0') + __characters;

      // Store all the characters, no matter whether they're part of the value or not. This is only safe if the buffer
      // we've been given is large enough.
      // TODO: Generate a mask on platforms which have native masked stores and use this code path unconditionally.
      __builtin_masked_store(__simd_vector<bool, 2 * sizeof(_Tp)>(true), __characters, __first);
      return {__first + __char_count, errc(0)};
    }
#endif

    ptrdiff_t __cap = __last - __first;
    int __n         = __width(__value);
    if (__n > __cap)
      return {__last, errc::value_too_large};

    __last             = __first + __n;
    char* __p          = __last;
    unsigned __divisor = 256;
    while (__value > __divisor) {
      unsigned __c = __value % __divisor;
      __value /= __divisor;
      __p -= 2;
      std::copy_n(&__base_16_lut[2 * __c], 2, __p);
    }
    if (__first != __last)
      do {
        unsigned __c = __value % 16;
        __value /= 16;
        *--__p = "0123456789abcdef"[__c];
      } while (__value != 0);
    return {__last, errc(0)};
  }
};

} // namespace __itoa

template <unsigned _Base, typename _Tp, __enable_if_t<(sizeof(_Tp) >= sizeof(unsigned)), int> = 0>
_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI int __to_chars_integral_width(_Tp __value) {
  return __itoa::__integral<_Base>::__width(__value);
}

template <unsigned _Base, typename _Tp, __enable_if_t<(sizeof(_Tp) < sizeof(unsigned)), int> = 0>
_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI int __to_chars_integral_width(_Tp __value) {
  return std::__to_chars_integral_width<_Base>(static_cast<unsigned>(__value));
}

template <unsigned _Base, typename _Tp, __enable_if_t<(sizeof(_Tp) >= sizeof(unsigned)), int> = 0>
_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI __to_chars_result
__to_chars_integral(char* __first, char* __last, _Tp __value) {
  return __itoa::__integral<_Base>::__to_chars(__first, __last, __value);
}

template <unsigned _Base, typename _Tp, __enable_if_t<(sizeof(_Tp) < sizeof(unsigned)), int> = 0>
_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI __to_chars_result
__to_chars_integral(char* __first, char* __last, _Tp __value) {
  return std::__to_chars_integral<_Base>(__first, __last, static_cast<unsigned>(__value));
}

template <typename _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI int __to_chars_integral_width(_Tp __value, unsigned __base) {
  _LIBCPP_ASSERT_INTERNAL(__value >= 0, "The function requires a non-negative value.");

  unsigned __base_2 = __base * __base;
  unsigned __base_3 = __base_2 * __base;
  unsigned __base_4 = __base_2 * __base_2;

  int __r = 0;
  while (true) {
    if (__value < __base)
      return __r + 1;
    if (__value < __base_2)
      return __r + 2;
    if (__value < __base_3)
      return __r + 3;
    if (__value < __base_4)
      return __r + 4;

    __value /= __base_4;
    __r += 4;
  }

  __libcpp_unreachable();
}

template <class _Tp, __enable_if_t<!is_signed<_Tp>::value, int> >
inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI __to_chars_result
__to_chars_integral(char* __first, char* __last, _Tp __value, int __base) {
  if (__base == 10) [[likely]]
    return std::__to_chars_itoa(__first, __last, __value, false_type());

  switch (__base) {
  case 2:
    return std::__to_chars_integral<2>(__first, __last, __value);
  case 8:
    return std::__to_chars_integral<8>(__first, __last, __value);
  case 16:
    return std::__to_chars_integral<16>(__first, __last, __value);
  }

  ptrdiff_t __cap = __last - __first;
  int __n         = std::__to_chars_integral_width(__value, __base);
  if (__n > __cap)
    return {__last, errc::value_too_large};

  __last    = __first + __n;
  char* __p = __last;
  do {
    unsigned __c = __value % __base;
    __value /= __base;
    *--__p = "0123456789abcdefghijklmnopqrstuvwxyz"[__c];
  } while (__value != 0);
  return {__last, errc(0)};
}

_LIBCPP_HIDE_FROM_ABI inline _LIBCPP_CONSTEXPR_SINCE_CXX14 char __hex_to_upper(char __c) {
  switch (__c) {
  case 'a':
    return 'A';
  case 'b':
    return 'B';
  case 'c':
    return 'C';
  case 'd':
    return 'D';
  case 'e':
    return 'E';
  case 'f':
    return 'F';
  }
  return __c;
}

#if _LIBCPP_STD_VER >= 17

to_chars_result to_chars(char*, char*, bool, int = 10) = delete;

template <typename _Tp, __enable_if_t<is_integral<_Tp>::value, int> = 0>
inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI to_chars_result
to_chars(char* __first, char* __last, _Tp __value) {
  using _Type = __make_32_64_or_128_bit_t<_Tp>;
  static_assert(!is_same<_Type, void>::value, "unsupported integral type used in to_chars");
  return std::__to_chars_itoa(__first, __last, static_cast<_Type>(__value), is_signed<_Tp>());
}

template <typename _Tp, __enable_if_t<is_integral<_Tp>::value, int> = 0>
inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI to_chars_result
to_chars(char* __first, char* __last, _Tp __value, int __base) {
  _LIBCPP_ASSERT_UNCATEGORIZED(2 <= __base && __base <= 36, "base not in [2, 36]");

  using _Type = __make_32_64_or_128_bit_t<_Tp>;
  return std::__to_chars_integral(__first, __last, static_cast<_Type>(__value), __base);
}

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CHARCONV_TO_CHARS_INTEGRAL_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: __builtin_ctzg is available since Clang 19 and GCC 14. When support for older versions is dropped, we can
//  refactor this code to exclusively use __builtin_ctzg.

#ifndef _LIBCPP___BIT_COUNTR_H
#define _LIBCPP___BIT_COUNTR_H

#include <__bit/rotate.h>
#include <__concepts/arithmetic.h>
#include <__config>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_unsigned.h>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR int __libcpp_ctz(unsigned __x) _NOEXCEPT {
  return __builtin_ctz(__x);
}

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR int __libcpp_ctz(unsigned long __x) _NOEXCEPT {
  return __builtin_ctzl(__x);
}

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR int __libcpp_ctz(unsigned long long __x) _NOEXCEPT {
  return __builtin_ctzll(__x);
}

#ifndef _LIBCPP_CXX03_LANG
// constexpr implementation for C++11 and later

// Precondition: __t != 0 (the caller __countr_zero handles __t == 0 as a special case)
template <class _Tp>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI constexpr int __countr_zero_impl(_Tp __t) _NOEXCEPT {
  static_assert(is_unsigned<_Tp>::value, "__countr_zero_impl only works with unsigned types");
  if constexpr (sizeof(_Tp) <= sizeof(unsigned int)) {
    return std::__libcpp_ctz(static_cast<unsigned int>(__t));
  } else if constexpr (sizeof(_Tp) <= sizeof(unsigned long)) {
    return std::__libcpp_ctz(static_cast<unsigned long>(__t));
  } else if constexpr (sizeof(_Tp) <= sizeof(unsigned long long)) {
    return std::__libcpp_ctz(static_cast<unsigned long long>(__t));
  } else {
#  if _LIBCPP_STD_VER == 11
    // A recursive constexpr implementation for C++11
    unsigned long long __ull       = static_cast<unsigned long long>(__t);
    const unsigned int __ulldigits = numeric_limits<unsigned long long>::digits;
    return __ull == 0ull ? __ulldigits + std::__countr_zero_impl<_Tp>(__t >> __ulldigits) : std::__libcpp_ctz(__ull);
#  else
    int __ret                      = 0;
    const unsigned int __ulldigits = numeric_limits<unsigned long long>::digits;
    while (static_cast<unsigned long long>(__t) == 0uLL) {
      __ret += __ulldigits;
      __t >>= __ulldigits;
    }
    return __ret + std::__libcpp_ctz(static_cast<unsigned long long>(__t));
#  endif
  }
}

#else
// implementation for C++03

template < class _Tp, __enable_if_t<is_unsigned<_Tp>::value && sizeof(_Tp) <= sizeof(unsigned int), int> = 0>
_LIBCPP_HIDE_FROM_ABI int __countr_zero_impl(_Tp __t) {
  return std::__libcpp_ctz(static_cast<unsigned int>(__t));
}

template < class _Tp,
           __enable_if_t<is_unsigned<_Tp>::value && (sizeof(_Tp) > sizeof(unsigned int)) &&
                             sizeof(_Tp) <= sizeof(unsigned long),
                         int> = 0 >
_LIBCPP_HIDE_FROM_ABI int __countr_zero_impl(_Tp __t) {
  return std::__libcpp_ctz(static_cast<unsigned long>(__t));
}

template < class _Tp,
           __enable_if_t<is_unsigned<_Tp>::value && (sizeof(_Tp) > sizeof(unsigned long)) &&
                             sizeof(_Tp) <= sizeof(unsigned long long),
                         int> = 0 >
_LIBCPP_HIDE_FROM_ABI int __countr_zero_impl(_Tp __t) {
  return std::__libcpp_ctz(static_cast<unsigned long long>(__t));
}

template < class _Tp, __enable_if_t<is_unsigned<_Tp>::value && (sizeof(_Tp) > sizeof(unsigned long long)), int> = 0 >
_LIBCPP_HIDE_FROM_ABI int __countr_zero_impl(_Tp __t) {
  int __ret                      = 0;
  const unsigned int __ulldigits = numeric_limits<unsigned long long>::digits;
  while (static_cast<unsigned long long>(__t) == 0uLL) {
    __ret += __ulldigits;
    __t >>= __ulldigits;
  }
  return __ret + std::__libcpp_ctz(static_cast<unsigned long long>(__t));
}

#endif // _LIBCPP_CXX03_LANG

template <class _Tp>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR int __countr_zero(_Tp __t) _NOEXCEPT {
  static_assert(is_unsigned<_Tp>::value, "__countr_zero only works with unsigned types");
#if __has_builtin(__builtin_ctzg)
  return __builtin_ctzg(__t, numeric_limits<_Tp>::digits);
#else
  return __t != 0 ? std::__countr_zero_impl(__t) : numeric_limits<_Tp>::digits;
#endif
}

#if _LIBCPP_STD_VER >= 20

template <__libcpp_unsigned_integer _Tp>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr int countr_zero(_Tp __t) noexcept {
  return std::__countr_zero(__t);
}

template <__libcpp_unsigned_integer _Tp>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr int countr_one(_Tp __t) noexcept {
  return __t != numeric_limits<_Tp>::max() ? std::countr_zero(static_cast<_Tp>(~__t)) : numeric_limits<_Tp>::digits;
}

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___BIT_COUNTR_H

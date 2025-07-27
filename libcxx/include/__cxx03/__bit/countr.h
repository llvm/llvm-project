//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: __builtin_ctzg is available since Clang 19 and GCC 14. When support for older versions is dropped, we can
//  refactor this code to exclusively use __builtin_ctzg.

#ifndef _LIBCPP___CXX03___BIT_COUNTR_H
#define _LIBCPP___CXX03___BIT_COUNTR_H

#include <__cxx03/__bit/rotate.h>
#include <__cxx03/__config>
#include <__cxx03/limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI int __libcpp_ctz(unsigned __x) _NOEXCEPT { return __builtin_ctz(__x); }

_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI int __libcpp_ctz(unsigned long __x) _NOEXCEPT {
  return __builtin_ctzl(__x);
}

_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI int __libcpp_ctz(unsigned long long __x) _NOEXCEPT {
  return __builtin_ctzll(__x);
}

template <class _Tp>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI int __countr_zero(_Tp __t) _NOEXCEPT {
#if __has_builtin(__builtin_ctzg)
  return __builtin_ctzg(__t, numeric_limits<_Tp>::digits);
#else  // __has_builtin(__builtin_ctzg)
  if (__t == 0)
    return numeric_limits<_Tp>::digits;
  if (sizeof(_Tp) <= sizeof(unsigned int))
    return std::__libcpp_ctz(static_cast<unsigned int>(__t));
  else if (sizeof(_Tp) <= sizeof(unsigned long))
    return std::__libcpp_ctz(static_cast<unsigned long>(__t));
  else if (sizeof(_Tp) <= sizeof(unsigned long long))
    return std::__libcpp_ctz(static_cast<unsigned long long>(__t));
  else {
    int __ret                      = 0;
    const unsigned int __ulldigits = numeric_limits<unsigned long long>::digits;
    while (static_cast<unsigned long long>(__t) == 0uLL) {
      __ret += __ulldigits;
      __t >>= __ulldigits;
    }
    return __ret + std::__libcpp_ctz(static_cast<unsigned long long>(__t));
  }
#endif // __has_builtin(__builtin_ctzg)
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___BIT_COUNTR_H

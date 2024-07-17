// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MATH_SPECIAL_FUNCTIONS_H
#define _LIBCPP___MATH_SPECIAL_FUNCTIONS_H

#include <__config>
#include <__math/abs.h>
#include <__math/copysign.h>
#include <__math/traits.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_integral.h>
#include <limits>
#include <stdexcept>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 17

template <class _Real>
_LIBCPP_HIDE_FROM_ABI _Real __assoc_laguerre(unsigned __n, unsigned __alpha, _Real __x) {
  // The associated/generalized Laguerre polynomial L_n^\alpha(x).
  // The implementation is based on the recurrence formula:
  //
  // (j+1) L_{j+1}^\alpha(x) = (-x + 2j + \alpha + 1) L_j^\alpha(x) - (j + \alpha) L_{j-1}^\alpha(x)
  //
  // Press, William H., et al. Numerical recipes 3rd edition: The art of scientific computing.
  // Cambridge university press, 2007, p. 183.

  // NOLINTBEGIN(readability-identifier-naming)
  if (__math::isnan(__x))
    return __x;

  if (__x < 0)
    std::__throw_domain_error("Argument `x` of Laguerre function is out of range: `x >= 0`.");

  _Real __L_0{1};
  if (__n == 0)
    return __L_0;

  _Real __L_n_prev = __L_0;
  _Real __L_n      = 1 + __alpha - __x;
  for (unsigned __i = 1; __i < __n; ++__i) {
    _Real __L_n_next =
        ((-__x + 2 * __i + __alpha + 1) * __L_n - (__i + __alpha) * __L_n_prev) / static_cast<_Real>(__i + 1);
    __L_n_prev = __L_n;
    __L_n      = __L_n_next;
  }

  if (!__math::isfinite(__L_n)) {
    // Overflow occured!
    // Can only happen for $x >> 1$ as _Real is at least double, and $__n < 128$ can be assumed.
    _Real __inf = std::numeric_limits<_Real>::infinity();
    return (__n & 1) ? -__inf : __inf;
  }

  return __L_n;
  // NOLINTEND(readability-identifier-naming)
}

template <class _Real>
_LIBCPP_HIDE_FROM_ABI _Real __laguerre(unsigned __n, _Real __x) {
  return std::__assoc_laguerre(__n, /*alpha=*/0, __x);
}

inline _LIBCPP_HIDE_FROM_ABI double laguerre(unsigned __n, double __x) { return std::__laguerre(__n, __x); }

inline _LIBCPP_HIDE_FROM_ABI float laguerre(unsigned __n, float __x) {
  // use double internally -- float is too prone to overflow!
  return static_cast<float>(std::laguerre(__n, static_cast<double>(__x)));
}

inline _LIBCPP_HIDE_FROM_ABI long double laguerre(unsigned __n, long double __x) { return std::__laguerre(__n, __x); }

inline _LIBCPP_HIDE_FROM_ABI float laguerref(unsigned __n, float __x) { return std::laguerre(__n, __x); }

inline _LIBCPP_HIDE_FROM_ABI long double laguerrel(unsigned __n, long double __x) { return std::laguerre(__n, __x); }

template <class _Integer, std::enable_if_t<std::is_integral_v<_Integer>, int> = 0>
_LIBCPP_HIDE_FROM_ABI double laguerre(unsigned __n, _Integer __x) {
  return std::laguerre(__n, static_cast<double>(__x));
}

inline _LIBCPP_HIDE_FROM_ABI double assoc_laguerre(unsigned __n, unsigned __m, double __x) {
  return std::__assoc_laguerre(__n, __m, __x);
}

inline _LIBCPP_HIDE_FROM_ABI float assoc_laguerre(unsigned __n, unsigned __m, float __x) {
  // use double internally -- float is too prone to overflow!
  return static_cast<float>(std::assoc_laguerre(__n, __m, static_cast<double>(__x)));
}

inline _LIBCPP_HIDE_FROM_ABI long double assoc_laguerre(unsigned __n, unsigned __m, long double __x) {
  return std::__assoc_laguerre(__n, __m, __x);
}

inline _LIBCPP_HIDE_FROM_ABI float assoc_laguerref(unsigned __n, unsigned __m, float __x) {
  return std::assoc_laguerre(__n, __m, __x);
}

inline _LIBCPP_HIDE_FROM_ABI long double assoc_laguerrel(unsigned __n, unsigned __m, long double __x) {
  return std::assoc_laguerre(__n, __m, __x);
}

template <class _Integer, std::enable_if_t<std::is_integral_v<_Integer>, int> = 0>
_LIBCPP_HIDE_FROM_ABI double assoc_laguerre(unsigned __n, unsigned __m, _Integer __x) {
  return std::assoc_laguerre(__n, __m, static_cast<double>(__x));
}

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MATH_SPECIAL_FUNCTIONS_H

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
#include <__math/roots.h>
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
_LIBCPP_HIDE_FROM_ABI _Real __assoc_legendre(unsigned __l, unsigned __m, _Real __x) {
  // The associated Legendre polynomial P_l^m(x).
  // The implementation is based on well-known recurrence formulas.
  // Resources:
  //    - Book: Press - Numerical Recipes - 3rd Edition - p. 294
  //    - Excellect write-up: https://justinwillmert.com/articles/2020/calculating-legendre-polynomials/

  // NOLINTBEGIN(readability-identifier-naming)
  if (__l < __m)
    return 0;
  /*
  // fixme: uncomment once `std::legendre` is integrated and git rebased.
  if (__m == 0)
    return std::legendre(__l, __x);
  */
  if (__math::isnan(__x))
    return __x;
  else if (__math::fabs(__x) > 1)
    std::__throw_domain_error("Argument `x` of associated Legendre function is out of range: `|x| <= 1`.");

  _Real __Pmm = 1; // init with P_0^0
  // Compute P_m^m: Explicit loop unrolling to work around computing square root.
  // Note: (1-x)*(1+x) is more accurate than (1-x*x)
  for (unsigned __n = 0; __n < __m / 2; ++__n)
    __Pmm *= (4 * __n + 1) * (4 * __n + 3) * (1 - __x) * (1 + __x);
  // Odd m case: Cannot combine two iterations. Thus, needs to compute square root.
  if (__m & 1)
    __Pmm *= (2 * __m - 1) * __math::sqrt((1 - __x) * (1 + __x));

  if (__l == __m)
    return __Pmm;

  // Factoring P^m_m out and multiplying it afterwards. Possible as recursion is linear.
  _Real __Pml      = __x * (2 * __m + 1) /* * __Pmm */; // init with P^m_{m+1}
  _Real __Pml_prev = 1 /* * __Pmm */;                   // init with P^m_m
  for (unsigned __n = __m + 2; __n <= __l; ++__n) {
    _Real __Pml_prev2 = __Pml_prev;                                                                       // P^m_{n-2}
    __Pml_prev        = __Pml;                                                                            // P^m_{n-1}
    __Pml             = (__x * (2 * __n - 1) * __Pml_prev - (__n + __m - 1) * __Pml_prev2) / (__n - __m); // P^m_n
  }
  return __Pml * __Pmm;
  // NOLINTEND(readability-identifier-naming)
}

inline _LIBCPP_HIDE_FROM_ABI double assoc_legendre(unsigned __l, unsigned __m, double __x) {
  return std::__assoc_legendre(__l, __m, __x);
}

inline _LIBCPP_HIDE_FROM_ABI float assoc_legendre(unsigned __l, unsigned __m, float __x) {
  return static_cast<float>(std::__assoc_legendre(__l, __m, static_cast<double>(__x)));
}

inline _LIBCPP_HIDE_FROM_ABI long double assoc_legendre(unsigned __l, unsigned __m, long double __x) {
  return std::__assoc_legendre(__l, __m, __x);
}

inline _LIBCPP_HIDE_FROM_ABI float assoc_legendref(unsigned __l, unsigned __m, float __x) {
  return std::assoc_legendre(__l, __m, __x);
}

inline _LIBCPP_HIDE_FROM_ABI long double assoc_legendrel(unsigned __l, unsigned __m, long double __x) {
  return std::assoc_legendre(__l, __m, __x);
}

template <class _Integer, std::enable_if_t<std::is_integral_v<_Integer>, int> = 0>
_LIBCPP_HIDE_FROM_ABI double assoc_legendre(unsigned __l, unsigned __m, _Integer __x) {
  return std::assoc_legendre(__l, __m, static_cast<double>(__x));
}

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MATH_SPECIAL_FUNCTIONS_H

// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the internal implementations of std::legendre
/// and std::assoc_legendre.
///
//===----------------------------------------------------------------------===//


#ifndef _LIBCPP_EXPERIMENTAL___MATH_LEGENDRE_H
#define _LIBCPP_EXPERIMENTAL___MATH_LEGENDRE_H

#include <experimental/__config>
#include <cmath>
#include <limits>
#include <stdexcept>

/// \return the Legendre polynomial \f$ P_{n}(x) \f$
/// \note The implementation is based on the recurrence formula
/// \f[
/// (n+1)P_{n+1}(x) = (2n+1)xP_{n}(x) - nP_{n-1}(x)
/// \f]
/// Press, William H., et al. Numerical recipes 3rd edition: The art of
/// scientific computing. Cambridge university press, 2007, p. 182.
template <class _Real>
_Real __libcpp_legendre_recurrence(unsigned __n, _Real __x) {
  if (__n == 0u)
    return _Real(1);

  _Real __t2(1);
  _Real __t1 = __x;
  for (unsigned __i = 1; __i < __n; ++__i) {
    const _Real __k = _Real(__i);
    _Real __t0 = ((_Real(2) * __k + _Real(1)) * __x * __t1 - __k * __t2) /
                 (__k + _Real(1));
    __t2 = __t1;
    __t1 = __t0;
  }
  return __t1;
}

template <class _Real> _Real __libcpp_legendre(unsigned __n, _Real __x) {
  if (std::isnan(__x))
    return std::numeric_limits<_Real>::quiet_NaN();

  if (std::abs(__x) > _Real(1))
    _VSTD::__throw_domain_error(
        "Argument of legendre function is out of range");

  return __libcpp_legendre_recurrence(__n, __x);
}

/// \return \f$ s^{-m} P_{l}^{m}(x) \f$ with an additonal scaling factor to
/// prevent overflow. \note The implementation is based on the recurrence
/// formula \f[ (l-m+1)P_{l+1}^{m}(x) = (2l+1)xP_{l}^{m}(x) -
/// (l+m)P_{l-1}^{m}(x) \f] with \f[ P_{m}^{m}(x) = \sqrt{1 -
/// x^2}^{m}\frac{(2m)!}{2^m m!} \f] and \f[ P_{m-1}^{m}(x) = 0 \f] \attention
/// The starting point of the recursion grows exponentially with __m! For large
/// m, we have the following relation: \f[ P_{m}^{m}(x) \approx \sqrt{1 -
/// x^2}^{m}\sqrt{2} 2^{n} \exp( n(\ln n - 1 )) \f] For example, for \f$ m = 40
/// \f$, we already have \f$ P_{40}^{40}(0) \approx 8 \cdot 10^{58}  \f$
/// \attention The so-called Condon-Shortley phase term is omitted in the C++17
/// standard's definition of std::assoc_laguerre.
template <class _Real>
_Real __libcpp_assoc_legendre_recurrence(unsigned __l, unsigned __m, _Real __x,
                                         _Real __scale = _Real(1)) {
  if (__m == 0u)
    return __libcpp_legendre_recurrence(__l, __x);

  if (__l < __m)
    return _Real(0);

  if (__l == 0u)
    return _Real(1);

  _Real __pmm = _Real(1);
  // Note: (1-x)*(1+x) is more accurate than (1-x*x)
  // "What Every Computer Scientist Should Know About Floating-Point
  // Arithmetic", David Goldberg, p. 38
  const _Real __t =
      std::sqrt((_Real(1) - __x) * (_Real(1) + __x)) / (_Real(2) * __scale);
  for (unsigned __i = 2u * __m; __i > __m; --__i)
    __pmm *= __t * __i;

  if (__l == __m)
    return __pmm;

  // Actually, we'd start with _pmm but it grows exponentially with __m.
  // Luckily, the recursion scales. So we can start with 1 and multiply
  // afterwards.
  _Real __t2 = _Real(1);
  _Real __t1 = _Real(2u * __m + 1u) * __x; // first iteration unfolded
  for (unsigned __i = __m + 1u; __i < __l; ++__i) {
    // As soon as one of the terms becomes inf, this will quickly lead to NaNs.
    // float just doesn't do it for the whole range up to l==127.
    const _Real __t0 =
        (_Real(2u * __i + 1u) * __x * __t1 - _Real(__i + __m) * __t2) /
        _Real(__i - __m + 1u);
    __t2 = __t1;
    __t1 = __t0;
  }
  return __t1 * __pmm;
}

template <class _Real>
_Real __libcpp_assoc_legendre(unsigned __n, unsigned __m, _Real __x) {
  if (std::isnan(__x))
    return std::numeric_limits<_Real>::quiet_NaN();

  if (std::abs(__x) > _Real(1))
    _VSTD::__throw_domain_error(
        "Argument of assoc_legendre function is out of range");

  return __libcpp_assoc_legendre_recurrence(__n, __m, __x);
}

#endif // _LIBCPP_EXPERIMENTAL___MATH_LEGENDRE_H

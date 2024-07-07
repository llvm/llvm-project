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
/// This file contains the internal implementations of std::hermite.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_EXPERIMENTAL___MATH_HERMITE_H
#define _LIBCPP_EXPERIMENTAL___MATH_HERMITE_H

#include <experimental/__config>
#include <cmath>
#include <limits>

/// \return the hermite polynomial \f$ H_{n}(x) \f$
/// \note The implementation is based on the recurrence formula
/// \f[
/// H_{n+1}(x) = 2x H_{n}(x) - 2 n H_{n-1}
/// \f]
/// Press, William H., et al. Numerical recipes 3rd edition: The art of
/// scientific computing. Cambridge university press, 2007, p. 183.
template <class _Real>
_Real __libcpp_hermite_recurrence(const unsigned __n, const _Real __x) {
  if (0 == __n)
    return 1;

  _Real __H_nPrev{1};
  _Real __H_n = 2 * __x;
  for (unsigned __i = 1; __i < __n; ++__i) {
    const _Real __H_nNext = 2 * (__x * __H_n - __i * __H_nPrev);
    __H_nPrev = __H_n;
    __H_n = __H_nNext;
  }
  return __H_n;
}

template <class _Real> _Real __libcpp_hermite(const unsigned __n, const _Real __x) {
  if (std::isnan(__x))
    return std::numeric_limits<_Real>::quiet_NaN();

  return __libcpp_hermite_recurrence(__n, __x);
}

#endif // _LIBCPP_EXPERIMENTAL___MATH_HERMITE_H

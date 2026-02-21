//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// test cases

#ifndef CASES_H
#define CASES_H

#include <cassert>
#include <complex>
#include <limits>
#include <type_traits>

#include "test_macros.h"

template <class T>
TEST_CONSTEXPR_CXX20 const std::complex<T> testcases[] = {
    std::complex<T>(+1.e-6, +1.e-6),
    std::complex<T>(-1.e-6, +1.e-6),
    std::complex<T>(-1.e-6, -1.e-6),
    std::complex<T>(+1.e-6, -1.e-6),

    std::complex<T>(+1.e+6, +1.e-6),
    std::complex<T>(-1.e+6, +1.e-6),
    std::complex<T>(-1.e+6, -1.e-6),
    std::complex<T>(+1.e+6, -1.e-6),

    std::complex<T>(+1.e-6, +1.e+6),
    std::complex<T>(-1.e-6, +1.e+6),
    std::complex<T>(-1.e-6, -1.e+6),
    std::complex<T>(+1.e-6, -1.e+6),

    std::complex<T>(+1.e+6, +1.e+6),
    std::complex<T>(-1.e+6, +1.e+6),
    std::complex<T>(-1.e+6, -1.e+6),
    std::complex<T>(+1.e+6, -1.e+6),

    std::complex<T>(-0.0, -1.e-6),
    std::complex<T>(-0.0, +1.e-6),
    std::complex<T>(-0.0, +1.e+6),
    std::complex<T>(-0.0, -1.e+6),
    std::complex<T>(+0.0, -1.e-6),
    std::complex<T>(+0.0, +1.e-6),
    std::complex<T>(+0.0, +1.e+6),
    std::complex<T>(+0.0, -1.e+6),

    std::complex<T>(-1.e-6, -0.0),
    std::complex<T>(+1.e-6, -0.0),
    std::complex<T>(+1.e+6, -0.0),
    std::complex<T>(-1.e+6, -0.0),
    std::complex<T>(-1.e-6, +0.0),
    std::complex<T>(+1.e-6, +0.0),
    std::complex<T>(+1.e+6, +0.0),
    std::complex<T>(-1.e+6, +0.0),

    std::complex<T>(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()),
    std::complex<T>(-std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN()),
    std::complex<T>(-2, std::numeric_limits<T>::quiet_NaN()),
    std::complex<T>(-1, std::numeric_limits<T>::quiet_NaN()),
    std::complex<T>(-0.5, std::numeric_limits<T>::quiet_NaN()),
    std::complex<T>(-0., std::numeric_limits<T>::quiet_NaN()),
    std::complex<T>(+0., std::numeric_limits<T>::quiet_NaN()),
    std::complex<T>(0.5, std::numeric_limits<T>::quiet_NaN()),
    std::complex<T>(1, std::numeric_limits<T>::quiet_NaN()),
    std::complex<T>(2, std::numeric_limits<T>::quiet_NaN()),
    std::complex<T>(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN()),

    std::complex<T>(std::numeric_limits<T>::quiet_NaN(), -std::numeric_limits<T>::infinity()),
    std::complex<T>(-std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity()),
    std::complex<T>(-2, -std::numeric_limits<T>::infinity()),
    std::complex<T>(-1, -std::numeric_limits<T>::infinity()),
    std::complex<T>(-0.5, -std::numeric_limits<T>::infinity()),
    std::complex<T>(-0., -std::numeric_limits<T>::infinity()),
    std::complex<T>(+0., -std::numeric_limits<T>::infinity()),
    std::complex<T>(0.5, -std::numeric_limits<T>::infinity()),
    std::complex<T>(1, -std::numeric_limits<T>::infinity()),
    std::complex<T>(2, -std::numeric_limits<T>::infinity()),
    std::complex<T>(std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity()),

    std::complex<T>(std::numeric_limits<T>::quiet_NaN(), -2),
    std::complex<T>(-std::numeric_limits<T>::infinity(), -2),
    std::complex<T>(-2, -2),
    std::complex<T>(-1, -2),
    std::complex<T>(-0.5, -2),
    std::complex<T>(-0., -2),
    std::complex<T>(+0., -2),
    std::complex<T>(0.5, -2),
    std::complex<T>(1, -2),
    std::complex<T>(2, -2),
    std::complex<T>(std::numeric_limits<T>::infinity(), -2),

    std::complex<T>(std::numeric_limits<T>::quiet_NaN(), -1),
    std::complex<T>(-std::numeric_limits<T>::infinity(), -1),
    std::complex<T>(-2, -1),
    std::complex<T>(-1, -1),
    std::complex<T>(-0.5, -1),
    std::complex<T>(-0., -1),
    std::complex<T>(+0., -1),
    std::complex<T>(0.5, -1),
    std::complex<T>(1, -1),
    std::complex<T>(2, -1),
    std::complex<T>(std::numeric_limits<T>::infinity(), -1),

    std::complex<T>(std::numeric_limits<T>::quiet_NaN(), -0.5),
    std::complex<T>(-std::numeric_limits<T>::infinity(), -0.5),
    std::complex<T>(-2, -0.5),
    std::complex<T>(-1, -0.5),
    std::complex<T>(-0.5, -0.5),
    std::complex<T>(-0., -0.5),
    std::complex<T>(+0., -0.5),
    std::complex<T>(0.5, -0.5),
    std::complex<T>(1, -0.5),
    std::complex<T>(2, -0.5),
    std::complex<T>(std::numeric_limits<T>::infinity(), -0.5),

    std::complex<T>(std::numeric_limits<T>::quiet_NaN(), -0.),
    std::complex<T>(-std::numeric_limits<T>::infinity(), -0.),
    std::complex<T>(-2, -0.),
    std::complex<T>(-1, -0.),
    std::complex<T>(-0.5, -0.),
    std::complex<T>(-0., -0.),
    std::complex<T>(+0., -0.),
    std::complex<T>(0.5, -0.),
    std::complex<T>(1, -0.),
    std::complex<T>(2, -0.),
    std::complex<T>(std::numeric_limits<T>::infinity(), -0.),

    std::complex<T>(std::numeric_limits<T>::quiet_NaN(), +0.),
    std::complex<T>(-std::numeric_limits<T>::infinity(), +0.),
    std::complex<T>(-2, +0.),
    std::complex<T>(-1, +0.),
    std::complex<T>(-0.5, +0.),
    std::complex<T>(-0., +0.),
    std::complex<T>(+0., +0.),
    std::complex<T>(0.5, +0.),
    std::complex<T>(1, +0.),
    std::complex<T>(2, +0.),
    std::complex<T>(std::numeric_limits<T>::infinity(), +0.),

    std::complex<T>(std::numeric_limits<T>::quiet_NaN(), 0.5),
    std::complex<T>(-std::numeric_limits<T>::infinity(), 0.5),
    std::complex<T>(-2, 0.5),
    std::complex<T>(-1, 0.5),
    std::complex<T>(-0.5, 0.5),
    std::complex<T>(-0., 0.5),
    std::complex<T>(+0., 0.5),
    std::complex<T>(0.5, 0.5),
    std::complex<T>(1, 0.5),
    std::complex<T>(2, 0.5),
    std::complex<T>(std::numeric_limits<T>::infinity(), 0.5),

    std::complex<T>(std::numeric_limits<T>::quiet_NaN(), 1),
    std::complex<T>(-std::numeric_limits<T>::infinity(), 1),
    std::complex<T>(-2, 1),
    std::complex<T>(-1, 1),
    std::complex<T>(-0.5, 1),
    std::complex<T>(-0., 1),
    std::complex<T>(+0., 1),
    std::complex<T>(0.5, 1),
    std::complex<T>(1, 1),
    std::complex<T>(2, 1),
    std::complex<T>(std::numeric_limits<T>::infinity(), 1),

    std::complex<T>(std::numeric_limits<T>::quiet_NaN(), 2),
    std::complex<T>(-std::numeric_limits<T>::infinity(), 2),
    std::complex<T>(-2, 2),
    std::complex<T>(-1, 2),
    std::complex<T>(-0.5, 2),
    std::complex<T>(-0., 2),
    std::complex<T>(+0., 2),
    std::complex<T>(0.5, 2),
    std::complex<T>(1, 2),
    std::complex<T>(2, 2),
    std::complex<T>(std::numeric_limits<T>::infinity(), 2),

    std::complex<T>(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity()),
    std::complex<T>(-std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity()),
    std::complex<T>(-2, std::numeric_limits<T>::infinity()),
    std::complex<T>(-1, std::numeric_limits<T>::infinity()),
    std::complex<T>(-0.5, std::numeric_limits<T>::infinity()),
    std::complex<T>(-0., std::numeric_limits<T>::infinity()),
    std::complex<T>(+0., std::numeric_limits<T>::infinity()),
    std::complex<T>(0.5, std::numeric_limits<T>::infinity()),
    std::complex<T>(1, std::numeric_limits<T>::infinity()),
    std::complex<T>(2, std::numeric_limits<T>::infinity()),
    std::complex<T>(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity()),
};

enum {zero, non_zero, inf, NaN, non_zero_nan};

template <class T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
TEST_CONSTEXPR_CXX20 bool test_isinf(T v) {
    return v == std::numeric_limits<T>::infinity() || v == -std::numeric_limits<T>::infinity();
}

template <class T, typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
TEST_CONSTEXPR_CXX20 bool test_isnan(T v) {
    return v != v;
}

template <class T>
TEST_CONSTEXPR_CXX20
int
classify(const std::complex<T>& x)
{
    if (x == std::complex<T>())
        return zero;
    if (test_isinf(x.real()) || test_isinf(x.imag()))
        return inf;
    if (test_isnan(x.real()) && test_isnan(x.imag()))
        return NaN;
    if (test_isnan(x.real()))
    {
        if (x.imag() == T(0))
            return NaN;
        return non_zero_nan;
    }
    if (test_isnan(x.imag()))
    {
        if (x.real() == T(0))
            return NaN;
        return non_zero_nan;
    }
    return non_zero;
}

template <class T>
inline int classify(T x) {
  if (x == 0)
    return zero;
  if (std::isinf(x))
    return inf;
  if (std::isnan(x))
    return NaN;
  return non_zero;
}

void is_about(float x, float y)
{
    assert(std::abs((x-y)/(x+y)) < 1.e-6);
}

void is_about(double x, double y)
{
    assert(std::abs((x-y)/(x+y)) < 1.e-14);
}

void is_about(long double x, long double y)
{
    assert(std::abs((x-y)/(x+y)) < 1.e-14);
}

#endif // CASES_H

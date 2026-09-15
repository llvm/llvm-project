//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SF_CMATH_COMMON_H
#define TEST_SF_CMATH_COMMON_H

#include <cassert>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <limits>

// std::type_identity is C++20 (we need to support C++17 here)
template <class T>
struct type_identity {
  typedef T type;
};
template <class T>
using type_identity_t = typename type_identity<T>::type;

template <class T>
bool between(type_identity_t<T> lower, T value, type_identity_t<T> upper) {
  return lower < value && value < upper;
}

// C 7.12.1/2 and /4, imported by [cmath.syn]/1: a domain error sets errno to EDOM and
// raises "invalid", a range error from overflow sets errno to ERANGE and raises
// "overflow" -- each channel only when math_errhandling advertises it.
//
// Note: math_errhandling must be queried with a runtime `if`, not `#if`. On Apple
// platforms it is defined as a function call (__math_errhandling()), so it is not a
// preprocessor constant and would break `#if` under -Wundef.
//
// The FE_* macros are optional (picolibc without hardware floating point does not define
// them), so every use is guarded. The flag is only asserted for the error cases: an
// error-free call may still raise "inexact" or "overflow" from an intermediate step, and
// a signaling NaN argument may raise "invalid" from a comparison.
template <class Func>
void check_no_domain_error(Func f) {
  if (math_errhandling & MATH_ERRNO)
    errno = EACCES;
  f();
  if (math_errhandling & MATH_ERRNO)
    assert(errno == EACCES);
}

template <class Func>
void check_domain_error(Func f) {
  if (math_errhandling & MATH_ERRNO)
    errno = EACCES;
#ifdef FE_INVALID
  if (math_errhandling & MATH_ERREXCEPT)
    std::feclearexcept(FE_INVALID);
#endif

  f();

  if (math_errhandling & MATH_ERRNO)
    assert(errno == EDOM);
#ifdef FE_INVALID
  if (math_errhandling & MATH_ERREXCEPT)
    assert(std::fetestexcept(FE_INVALID) != 0);
#endif
}

template <class Func>
void check_range_error(Func f) {
  if (math_errhandling & MATH_ERRNO)
    errno = EACCES;
#ifdef FE_OVERFLOW
  if (math_errhandling & MATH_ERREXCEPT)
    std::feclearexcept(FE_OVERFLOW);
#endif

  f();

  if (math_errhandling & MATH_ERRNO)
    assert(errno == ERANGE);
#ifdef FE_OVERFLOW
  if (math_errhandling & MATH_ERREXCEPT)
    assert(std::fetestexcept(FE_OVERFLOW) != 0);
#endif
}

// Distance between `value` and `expected` in units of the last place of `expected`.
//
// Not a bit comparison: `long double` is not portably bit-castable. On x86-64 it occupies
// 128 bits of which only 80 carry the value, and the padding is indeterminate, so two equal
// values need not have equal object representations; std::bit_cast also needs C++20. Not a
// std::nextafter loop either -- that costs an iteration per ulp. std::frexp gives the
// exponent of the binade holding `expected`, whose last place is 2^(exponent - digits).
//
// Immediately below a power of two the spacing halves, so a step of one place there is
// reported as half of one. The tolerances allow for it.
//
// A NaN or infinite result propagates through the subtraction, so it compares false against
// any tolerance and the caller's assert fires, which is what a broken result should do.
template <class Float>
Float ulp_distance(Float value, type_identity_t<Float> expected) {
  assert(std::isfinite(expected) && expected != 0);
  int exponent = 0;
  std::frexp(expected, &exponent);
  const Float last_place = std::ldexp(Float(1), exponent - std::numeric_limits<Float>::digits);
  return std::abs(value - expected) / last_place;
}

#endif // TEST_SF_CMATH_COMMON_H

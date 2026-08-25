//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Shared body for the assoc_laguerre error-reporting tests, which run it under several
// math-related compile flags.
//
// C 7.12.1/2 and /4, imported by [cmath.syn]/1, tie the reporting channel to
// math_errhandling and leave the value returned on a domain error implementation-defined.
// libc++ returns a quiet NaN and reports on *both* channels unconditionally, because
// math_errhandling describes the caller's translation unit while these functions are
// defined out of line in the built library: a caller compiled with -fno-math-errno (glibc
// then defines math_errhandling as MATH_ERREXCEPT) or with -ffast-math (math_errhandling
// becomes 0) must still see errno and the flags set. That is what these tests pin down,
// so the assertions here are deliberately not guarded on math_errhandling.

#ifndef TEST_LIBCXX_SF_CMATH_ERROR_REPORTING_H
#define TEST_LIBCXX_SF_CMATH_ERROR_REPORTING_H

#include <cassert>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <limits>

// -ffast-math implies -ffinite-math-only, which lets the compiler fold isnan and a
// comparison against an infinity to a constant. Only the reporting channels are checked
// there; the returned values are covered by the test in test/std.
#ifdef __FAST_MATH__
#  define TEST_SF_CHECK_VALUES 0
#else
#  define TEST_SF_CHECK_VALUES 1
#endif

// The FE_* macros are optional (picolibc without hardware floating point does not define
// them), so every use is guarded.
template <class Func, class Float>
void test_domain_error(Func assoc_laguerre, Float x) {
  errno = 0;
#ifdef FE_INVALID
  std::feclearexcept(FE_INVALID);
#endif

  [[maybe_unused]] Float result = assoc_laguerre(1, 0, x);

  assert(errno == EDOM);
#ifdef FE_INVALID
  assert(std::fetestexcept(FE_INVALID) != 0);
#endif
#if TEST_SF_CHECK_VALUES
  assert(std::isnan(result));
#endif
}

template <class Func, class Float>
void test_range_error(Func assoc_laguerre, Float x) {
  errno = 0;
#ifdef FE_OVERFLOW
  std::feclearexcept(FE_OVERFLOW);
#endif

  // L^0_2(x) = 1 - 2x + x^2/2 does not fit for an x near the largest finite value
  [[maybe_unused]] Float result = assoc_laguerre(2, 0, x);

  assert(errno == ERANGE);
#ifdef FE_OVERFLOW
  assert(std::fetestexcept(FE_OVERFLOW) != 0);
#endif
#if TEST_SF_CHECK_VALUES
  assert(result == std::numeric_limits<Float>::infinity());
#endif
}

inline void test_error_reporting() {
  auto laguerre_f  = [](unsigned n, unsigned m, float x) { return std::assoc_laguerref(n, m, x); };
  auto laguerre    = [](unsigned n, unsigned m, double x) { return std::assoc_laguerre(n, m, x); };
  auto laguerre_l  = [](unsigned n, unsigned m, long double x) { return std::assoc_laguerrel(n, m, x); };
  auto laguerre_ff = [](unsigned n, unsigned m, float x) { return std::assoc_laguerre(n, m, x); };
  auto laguerre_ll = [](unsigned n, unsigned m, long double x) { return std::assoc_laguerre(n, m, x); };

  test_domain_error(laguerre_f, -1.0f);
  test_domain_error(laguerre_ff, -1.0f);
  test_domain_error(laguerre, -1.0);
  test_domain_error(laguerre_l, -1.0L);
  test_domain_error(laguerre_ll, -1.0L);

  // -inf is outside the x >= 0 domain too ([sf.cmath.general]/2)
  if (std::numeric_limits<double>::has_infinity)
    test_domain_error(laguerre, -std::numeric_limits<double>::infinity());

  test_range_error(laguerre_f, std::numeric_limits<float>::max());
  test_range_error(laguerre_ff, std::numeric_limits<float>::max());
  test_range_error(laguerre, std::numeric_limits<double>::max());
  test_range_error(laguerre_l, std::numeric_limits<long double>::max());
  test_range_error(laguerre_ll, std::numeric_limits<long double>::max());
}

#endif // TEST_LIBCXX_SF_CMATH_ERROR_REPORTING_H

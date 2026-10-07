//===-- include/flang/Common/erfc-scaled-accurate.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements ERFC_SCALED (erfcx) to nearly full precision of the argument
// type, templated over both the type and the math functions it needs, so the
// same algorithm can serve compile-time constant folding (see
// flang/lib/Evaluate/intrinsics-library.cpp) and the runtime (flang-rt), and
// types with no <cmath> overloads (__float128 via libquadmath) alongside
// float, double, and long double.
//
// The algorithm was contributed in
// https://github.com/llvm/llvm-project/pull/219697 for binary128; this header
// generalizes it per type:
//
//   x >= threshold:  the asymptotic expansion
//                    erfcx(x) ~ 1/(x*sqrt(pi)) * (1 - 1/(2x^2) + 3/(4x^4) -
//                    ...) truncated after kTerms terms. Host-libm-free, so
//                    results are reproducible across hosts with the same IEEE
//                    format, rounding mode, and no FP contraction.
//   0 <= x < threshold:  exp(x*x)*erfc(x) directly, with the squaring error
//                    compensated: u = x*x, e = fma(x,x,-u) (the exact low
//                    part), and exp(u+e) expanded as exp(u)*(1+e). Without
//                    the compensation, exp() amplifies the squaring rounding
//                    error by a factor of x*x (hundreds of ulps near the
//                    threshold).
//   x < 0:           the reflection erfc(-x) = 2 - erfc(x), so
//                    erfcx(x) = 2*exp(x*x) - erfcx(-x), which grows and
//                    overflows to +infinity where 2*exp(x*x) does.
//
// Two constants deserve their shared story told once:
//
// * The direct/series threshold is per-type: 16 in general, 9 for float.
//   The series needs x large enough for kTerms truncation to be accurate
//   (with 22 terms: ~3.1e5 binary128 eps at x = 12, <= 0.59 eps at x = 16, and
//   below one eps above; at float's 24-bit precision, x = 9 is already exact
//   to the last ulp). The direct form needs exp(x*x) finite and erfc(x)
//   normal: for binary128 that holds through the threshold with a wide
//   margin, but expf overflows at x*x > 88.72 (x ~ 9.42) and erfcf is
//   subnormal there and exactly zero by x ~ 10 - a threshold of 16 would
//   return Inf then NaN over [9.42, 16) at float. Threshold 9 keeps float on
//   the series exactly where its direct form breaks down.
//
// * The negative-branch cutoff 107 is one universal constant. It is sized
//   for binary128, where the true value first exceeds the format's range at
//   x*x > 16384*ln2 (|x| ~ 106.567): returning +Inf from |x| >= 107 on is
//   correct there, and the cutoff must not be larger than ~1.09e2466, where
//   x*x itself overflows and the compensation term e = fma(x,x,-Inf) becomes
//   -Inf, turning exp(u)*(1+e) into Inf*(-Inf) = -Inf - an infinity of the
//   wrong sign. For narrower types the same constant is safe without
//   adjustment: their 2*exp(x*x) overflows to +Inf - the correctly rounded
//   answer - far below 107 (x ~ 9.4 for float, ~26.63 for double), the
//   per-type threshold routes the subtrahend erfcx(-x) to the never-
//   overflowing series, and x*x stays finite below 107 in every supported
//   format, so the fma hazard is unreachable.
//
// Do not reach for std::numeric_limits<T> here beyond the default policy:
// libstdc++ leaves numeric_limits<__float128> unspecialized (all members
// zero) while libc++ and MSVC differ again, so any use of it in the shared
// code would compile everywhere and be wrong somewhere. Everything
// type-specific comes in through the policy.

#ifndef FORTRAN_COMMON_ERFC_SCALED_ACCURATE_H_
#define FORTRAN_COMMON_ERFC_SCALED_ACCURATE_H_

#include "flang/Common/api-attrs.h"
#include <cmath>
#include <limits>
#include <type_traits>

namespace Fortran::common {

// Default math policy for types <cmath> covers. Consumers whose type has no
// std:: overloads (e.g. __float128) supply their own policy with the same
// four members instead of specializing anything.
template <typename T> struct ErfcScaledStdHostPolicy {
  static inline RT_API_ATTRS T Exp(T x) { return std::exp(x); }
  static inline RT_API_ATTRS T Erfc(T x) { return std::erfc(x); }
  static inline RT_API_ATTRS T Fma(T x, T y, T z) { return std::fma(x, y, z); }
  static inline RT_API_ATTRS T Infinity() {
    return std::numeric_limits<T>::infinity();
  }
};

// sqrt(pi) as the sum of three exactly representable doubles: correctly
// rounded up to and including binary128 with no long double literal capping
// the precision and no Q suffix (a GNU extension). Keep the grouping; do not
// fold the sum through a narrower type.
template <typename T> constexpr RT_API_ATTRS T ErfcScaledSqrtPi() {
  return static_cast<T>(1.772453850905516) +
      static_cast<T>(-7.666586499825799e-17) +
      static_cast<T>(-1.3058334907945429e-33);
}

// exp(x*x) with the squaring rounding error compensated (see file comment).
template <typename T, typename P = ErfcScaledStdHostPolicy<T>>
inline RT_API_ATTRS T ErfcScaledExpOfSquare(T x) {
  T u{x * x};
  T e{P::Fma(x, x, -u)};
  return P::Exp(u) * (T{1} + e);
}

template <typename T, typename P = ErfcScaledStdHostPolicy<T>>
inline RT_API_ATTRS T ErfcScaledPositive(T x) {
  // Per-type direct/series threshold; rationale in the file comment.
  constexpr T threshold{std::is_same_v<T, float> ? T{9} : T{16}};
  if (x < threshold) {
    return ErfcScaledExpOfSquare<T, P>(x) * P::Erfc(x);
  }
  // Asymptotic series, truncated after kTerms terms. Truncation error at the
  // binary128 threshold x = 16 measures <= 0.59 eps and shrinks above it;
  // do not read the count as giving correctly rounded results.
  constexpr int kTerms{22};
  const T inv2x2{T{1} / (T{2} * x * x)};
  T term{1};
  T sum{term};
  for (int k{1}; k < kTerms; ++k) {
    term *= -static_cast<T>(2 * k - 1) * inv2x2;
    sum += term;
  }
  // Divide by sqrt(pi) before dividing by x: x*sqrt(pi) can overflow near
  // the top of the format (x >~ 6.7e4931 in binary128) where the true result
  // is a representable subnormal.
  return (sum / ErfcScaledSqrtPi<T>()) / x;
}

template <typename T, typename P = ErfcScaledStdHostPolicy<T>>
inline RT_API_ATTRS T ErfcScaledAccurate(T x) {
  if (x < T{0}) {
    T ax{-x};
    if (ax >= T{107}) { // universal cutoff; rationale in the file comment
      return P::Infinity();
    }
    // erfc(-x) = 2 - erfc(x), so erfcx(-ax) = 2*exp(ax*ax) - erfcx(ax).
    // No catastrophic cancellation: the minuend 2*exp(ax*ax) >= 2 while the
    // subtrahend erfcx(ax) <= 1.
    return T{2} * ErfcScaledExpOfSquare<T, P>(ax) -
        ErfcScaledPositive<T, P>(ax);
  }
  return ErfcScaledPositive<T, P>(x);
}

} // namespace Fortran::common
#endif // FORTRAN_COMMON_ERFC_SCALED_ACCURATE_H_

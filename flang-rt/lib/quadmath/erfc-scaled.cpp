//===-- lib/quadmath/erfc-scaled.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ERFC_SCALED at REAL(16).
//
// The other kinds share a Cody rational approximation in
// flang/include/flang/Common/erfc-scaled.h. It is not used here, deliberately.
// Its coefficient arrays carry about seventeen significant digits and its
// sqrtpi/rsqrtpi are long double literals, so evaluated at __float128 it
// returns a plausible answer that is wrong by up to 2.8e-17 relative - some
// 1.5e17 epsilons of binary128, which is the accuracy of double precision
// delivered in a container that promises thirty-three digits. That was measured
// against mpmath at 50 decimal places before this file was written; gfortran on
// the same points is correctly rounded to within 0.91 eps.
//
// This uses the exact entry points the quadmath build already provides.
//
//===----------------------------------------------------------------------===//

#include "math-entries.h"

namespace Fortran::runtime {
extern "C" {

#if HAS_LDBL128 || HAS_FLOAT128

using F128Ty = CppTypeFor<TypeCategory::Real, 16>;

// Where the two branches meet.
//
// Below it, ERFC_SCALED is exp(x*x)*erfc(x) evaluated directly, which is only
// possible while neither factor leaves the format: exp(x*x) overflows binary128
// above x = 106.567 (x*x > 16384*ln2) and erfc underflows above x = 106.536.
//
// Above it, the asymptotic expansion is used. That reaches one epsilon of
// binary128 from about x = 12 upwards - 1.4e-6 eps at x = 12, but still 1.3e8
// eps at x = 8 - so it must not be used below there.
//
// The usable window is therefore [12, 106.5], and sixteen sits with margin on
// both sides: the series is already good to 0.59 eps there, and the direct form
// is a factor of six short of overflowing.
static constexpr F128Ty kThreshold{16};

// Terms of the asymptotic series. Twenty-two are needed at the threshold; the
// series does not begin to diverge until k ~ x*x = 256, so the same count is
// safe everywhere above it and only grows more accurate.
static constexpr int kTerms{22};

// sqrt(pi), as the sum of three doubles.
//
// Not one literal: an unsuffixed literal is a double, and an L suffix would cap
// this at the 64-bit mantissa of x86-64 long double and put a 1e-19 floor under
// every result. A Q suffix would be wrong when this type is long double. Three
// exactly representable doubles sum to sqrt(pi) within 7.6e-17 eps of binary128
// and need no suffix at all.
static constexpr F128Ty kSqrtPi{static_cast<F128Ty>(1.772453850905516) +
    static_cast<F128Ty>(-7.666586499825799e-17) +
    static_cast<F128Ty>(-1.3058334907945429e-33)};

// exp() amplifies the rounding error of a squaring by a factor of x*x - some
// 250 epsilons near the threshold. So the square is taken exactly: x*x = u + e
// with e the FMA residual, and exp(u + e) = exp(u)*exp(e) is expanded as
// exp(u)*(1 + e), which suffices because e is below 1e-30 here and the dropped
// e*e/2 is below 1e-60.
//
// Without this the error reaches 123 eps on arguments whose square is not
// exactly representable - and stays near 0.4 eps on arguments whose square is,
// which is exactly why an accuracy test must not step by 0.5.
static F128Ty ScaledExpOfSquare(F128Ty x) {
  F128Ty u{x * x};
  F128Ty e{Fma<true>::invoke(x, x, -u)};
  return Exp<true>::invoke(u) * (1 + e);
}

static F128Ty ErfcScaledPositive(F128Ty x) {
  if (x < kThreshold) {
    return ScaledExpOfSquare(x) * Erfc<true>::invoke(x);
  }
  // 1/(x*sqrt(pi)) * (1 - 1/(2x^2) + 3/(4x^4) - 15/(8x^6) + ...)
  F128Ty inv2x2{1 / (2 * x * x)};
  F128Ty sum{1}, term{1};
  for (int k{1}; k <= kTerms; ++k) {
    term *= -static_cast<F128Ty>(2 * k - 1) * inv2x2;
    sum += term;
  }
  return sum / (x * kSqrtPi);
}

F128Ty RTDEF(ErfcScaled16)(F128Ty x) {
  if (x >= 0) {
    return ErfcScaledPositive(x);
  }
  // erfc(-x) = 2 - erfc(x), so ERFC_SCALED(-x) = 2*exp(x*x) - ERFC_SCALED(x).
  // The asymptotic branch describes the decaying tail only and must never see a
  // negative argument; this side grows instead, and leaves the format at the
  // same place the direct branch does.
  F128Ty ax{-x};
  if (ax >= 106) {
    return F128_RT_INFINITY;
  }
  return 2 * ScaledExpOfSquare(ax) - ErfcScaledPositive(ax);
}
#endif

} // extern "C"
} // namespace Fortran::runtime

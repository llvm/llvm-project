//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the bfloat16 tgamma function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_TGAMMABF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_TGAMMABF16_H

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/expf.h"
#include "src/__support/math/sinpif.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {
namespace tgammabf16_internal {

// The hexadecimal literals were generated with sollya:
// > display = hexadecimal;
// > round(pi, D, RN);
// > round(log(sqrt(2 * pi)), D, RN);
// > round(log(2), D, RN);
// > round(sqrt(2), D, RN);
constexpr double PI = 0x1.921fb54442d18p1;
constexpr double LOG_SQRT_2_PI = 0x1.d67f1c864beb5p-1;
constexpr double LOG_2 = 0x1.62e42fefa39efp-1;
constexpr double SQRT_2 = 0x1.6a09e667f3bcdp0;

// EULER_GAMMA is the Euler-Mascheroni constant. Its correctly rounded
// binary64 value was generated with MPFR's mpfr_const_euler.
constexpr double EULER_GAMMA = 0x1.2788cfc6fb619p-1;

// Paul Godfrey's exact Lanczos approximation coefficients (g=7, n=9)
// Reference: "A note on the computation of the convergent Lanczos complex
// Gamma approximation" by Paul Godfrey (2001).
// Original:
// https://web.archive.org/web/20060915161115/http://my.fit.edu/~gabdo/gamma.txt
// Mirror: http://www.mrob.com/pub/ries/lanczos-gamma.html
constexpr double LANCZOS_COEFFS[9] = {
    0x1.ffffffffff950p-1,  0x1.52429b6c30b05p+9,  -0x1.3ac8e8ed4171bp+10,
    0x1.81a9661d3b4d8p+9,  -0x1.613ae51a32f5dp+7, 0x1.903c27f8b9c81p+3,
    -0x1.1bcb2992b2855p-3, 0x1.4f0514e4e324fp-17, 0x1.435508f3faeefp-23};

// Sets ERANGE, raises FE_OVERFLOW, and returns the overflow value
// (+inf or max_normal) according to the sign and active IEEE 754 rounding mode.
LIBC_INLINE bfloat16 tgammabf16_overflow(Sign sign) {
  fputil::set_errno_if_required(ERANGE);
  fputil::raise_except_if_required(FE_OVERFLOW);

#ifdef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  return fputil::FPBits<bfloat16>::inf(sign).get_val();
#else
  switch (fputil::quick_get_round()) {
  case FE_TONEAREST:
    return fputil::FPBits<bfloat16>::inf(sign).get_val();
  case FE_UPWARD:
    return sign == Sign::POS
               ? fputil::FPBits<bfloat16>::inf(Sign::POS).get_val()
               : fputil::FPBits<bfloat16>::max_normal(Sign::NEG).get_val();
  case FE_DOWNWARD:
    return sign == Sign::POS
               ? fputil::FPBits<bfloat16>::max_normal(Sign::POS).get_val()
               : fputil::FPBits<bfloat16>::inf(Sign::NEG).get_val();
  case FE_TOWARDZERO:
    return fputil::FPBits<bfloat16>::max_normal(sign).get_val();
  default:
    return fputil::FPBits<bfloat16>::max_normal(sign).get_val();
  }
#endif
}

// This is only called with x in [7.5, 135.5] or with the positive Lanczos
// coefficient sum. Range reduction into [sqrt(2)/2, sqrt(2)] confines
// r = (m - 1) / (m + 1) to [-0.172, 0.172], making the degree-13 atanh Taylor
// series accurate to ~2^-41 (sufficient for bfloat16 requirements).
LIBC_INLINE double tgammabf16_log(double x) {
  using DoubleBits = fputil::FPBits<double>;
  DoubleBits xbits(x);
  int exponent =
      static_cast<int>(xbits.get_biased_exponent()) - DoubleBits::EXP_BIAS;
  xbits.set_biased_exponent(DoubleBits::EXP_BIAS);
  double mantissa = xbits.get_val();
  if (mantissa > SQRT_2) {
    mantissa *= 0.5;
    ++exponent;
  }

  double r = (mantissa - 1.0) / (mantissa + 1.0);
  double r2 = r * r;
  // Maclaurin series: log(mantissa) = log((1 + r) / (1 - r)) = 2 * atanh(r)
  //                                 = 2 * r * (1 + r^2/3 + r^4/5 + ...).
  // The coefficients 1/(2k + 1) for k = 1..6 were generated with Sollya:
  // > for k from 1 to 6 do print(round(1 / (2 * k + 1), D, RN));
  double p = fputil::polyeval(r2, 0x1.5555555555555p-2, 0x1.999999999999ap-3,
                              0x1.2492492492492p-3, 0x1.c71c71c71c71cp-4,
                              0x1.745d1745d1746p-4, 0x1.3b13b13b13b14p-4);
  return fputil::multiply_add(static_cast<double>(exponent), LOG_2,
                              2.0 * r * fputil::multiply_add(r2, p, 1.0));
}

} // namespace tgammabf16_internal

LIBC_INLINE bfloat16 tgammabf16(bfloat16 x) {
  using FPBits = fputil::FPBits<bfloat16>;

  FPBits xbits(x);
  uint16_t x_abs = xbits.uintval() & 0x7fffU;

  if (LIBC_UNLIKELY(x_abs >= FPBits::EXP_MASK)) {
    if (xbits.is_nan()) {
      if (xbits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return x;
    }
    if (xbits.is_pos())
      return x;
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  if (LIBC_UNLIKELY(xbits.is_zero())) {
    fputil::set_errno_if_required(ERANGE);
    fputil::raise_except_if_required(FE_DIVBYZERO);
    return FPBits::inf(xbits.sign()).get_val();
  }

  if (LIBC_UNLIKELY(xbits.is_neg())) {
    int biased_exp = x_abs >> FPBits::FRACTION_LEN;
    if (biased_exp >= FPBits::EXP_BIAS) {
      int e = biased_exp - FPBits::EXP_BIAS;
      if (e >= FPBits::FRACTION_LEN ||
          (xbits.get_mantissa() &
           static_cast<uint16_t>((1U << (FPBits::FRACTION_LEN - e)) - 1U)) ==
              0U) {
        fputil::set_errno_if_required(EDOM);
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
    }
  }

  float xf = static_cast<float>(x);

  // Gamma is strictly increasing for x >= 2, and Gamma(35) is finite in
  // bfloat16 while every larger positive bfloat16 input overflows.
  if (LIBC_UNLIKELY(xf > 35.0f))
    return tgammabf16_internal::tgammabf16_overflow(Sign::POS);

  // Fast path for exact positive integers
  if (xf > 0.0f && xf <= 35.0f) {
    int n = static_cast<int>(xf);
    if (xf == static_cast<float>(n)) {
      double res = 1.0;
      for (int i = 1; i < n; ++i)
        res *= i;
      return fputil::cast<bfloat16>(res);
    }
  }

  bool reflection = xbits.is_neg();
  bool divide_by_x = false;

  double xd = static_cast<double>(xf);
  double x_eval = xd;
  double res;

  // Fast path for tiny positive inputs to prevent exact-boundary overshoots.
  if (LIBC_UNLIKELY(xf > 0.0f && xf < 0x1.0p-8f)) {
    res = (1.0 / xd) - tgammabf16_internal::EULER_GAMMA;
  } else {
    if (reflection) {
      x_eval = 1.0 - xd;
    } else if (xf < 1.0f) {
      divide_by_x = true;
      x_eval = xd + 1.0;
    }

    double z = x_eval - 1.0;
    double a = tgammabf16_internal::LANCZOS_COEFFS[0];
    for (int i = 1; i < 9; ++i) {
      a +=
          tgammabf16_internal::LANCZOS_COEFFS[i] / (z + static_cast<double>(i));
    }

    double t = z + 7.5;

    double log_t = tgammabf16_internal::tgammabf16_log(t);
    double base_term = (tgammabf16_internal::LOG_SQRT_2_PI - t) +
                       tgammabf16_internal::tgammabf16_log(a);
    double exp_arg = fputil::multiply_add(z + 0.5, log_t, base_term);

    if (reflection) {
      double sin_pi_x = static_cast<double>(math::sinpif(xf));
      if (LIBC_UNLIKELY(sin_pi_x == 0.0)) {
        fputil::set_errno_if_required(ERANGE);
        fputil::raise_except_if_required(FE_UNDERFLOW);
        return FPBits::zero(Sign::POS).get_val();
      }

      // Gamma(x_eval) can exceed the float range on the reflection path.
      // Evaluate 1 / Gamma(x_eval) directly instead.
      res = (tgammabf16_internal::PI / sin_pi_x) *
            static_cast<double>(math::expf(static_cast<float>(-exp_arg)));
    } else {
      // Since x <= 35, exp_arg < 88.0, which is within normal float bounds
      res = static_cast<double>(math::expf(static_cast<float>(exp_arg)));
      if (divide_by_x)
        res = res / xd;
    }
  }

  double abs_res = res < 0.0 ? -res : res;

  // 0x1.ffp127 is max normal + 0.5 ULP in bfloat16, the exact RTN overflow
  // threshold
  if (LIBC_UNLIKELY(abs_res > 0x1.ffp+127)) {
    return tgammabf16_internal::tgammabf16_overflow(res > 0.0 ? Sign::POS
                                                              : Sign::NEG);
  }

  return fputil::cast<bfloat16>(res);
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_TGAMMABF16_H

//===-- Implementation header for lgammabf16 --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_LGAMMABF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_LGAMMABF16_H

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/log.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {
namespace lgammabf16_internal {

// lgamma_positive_d: compute lgamma(x) for x > 0, returning double.
//
// Takes double so callers can pass (1.0 + ax) without float precision loss.
// The double return is necessary to avoid double-rounding: the final
// fputil::cast<bfloat16> needs the full double precision so it can correctly
// break ties (see the 0x35E5 example in lgammabf16 below).
//
// For x < 4, applies the recurrence lgamma(x) = lgamma(x+1) - ln(x) until
// x reaches [4, 8), then evaluates the polynomial. This is critical because
// the [2,3) polynomial has max_rel_err=9.98e-6 which, near its edges (t near
// +0.5 or -0.5), causes ~6e-6 absolute error. After subtracting ln(x), the
// result near x=1 or x=2 can be as small as ~0.002, giving ~0.45 ULP error --
// which fails the directed-rounding tolerance test. Polynomials for [4,5) and
// above have max_rel_err <= 6.61e-7, keeping final error well under 0.1 ULP.
//
LIBC_INLINE double lgamma_positive_d(double x) {
  // Coefficients for lgamma on [n, n+1), centered at n+0.5.
  // Each row: {c0, c1, c2, c3, c4} for Estrin evaluation
  // where t = x - (n + 0.5), n = 1..7.
  // P is a degree-4 fit to lgamma(t + n+0.5) on t in [-0.5, 0.5].
  // For n = 3..7: mpmath.chebyfit (no native lgamma in Sollya):
  //   > mpmath.mp.dps = 50
  //   > f = lambda t: mpmath.loggamma(t + (n + 0.5))
  //   > coeffs = mpmath.chebyfit(f, [-0.5, 0.5], 5)
  // For n = 1, 2: chebyfit's relative error is too poor near the
  // zeros of lgamma at x=1 and x=2 (see fit_irls_relative_error()),
  // so these two rows instead use an IRLS (Lawson-iteration) fit that
  // directly minimizes relative error rather than plain interpolation.
  // Reversed to ascending [c0..c4] and rounded to float32 for Estrin.
  constexpr float LGAMMA_POLY[7][5] = {
      // [1,2), center=1.5, max_rel_err=1.95e-03 (IRLS relative-error fit)
      {-0x1.ee9a58p-4f, 0x1.36b582p-5f, 0x1.dd697ep-2f, -0x1.36ad56p-3f,
       0x1.130d8p-4f},
      // [2,3), center=2.5, max_rel_err=4.32e-05 (IRLS relative-error fit)
      {0x1.2386f2p-2f, 0x1.6803b6p-1f, 0x1.f53b64p-3f, -0x1.46e712p-5f,
       0x1.736f46p-7f},
      // [3,4), center=3.5, max_rel_err=1.79e-06 (chebyfit)
      {0x1.337302p+0f, 0x1.1a6936p+0f, 0x1.52480cp-3f, -0x1.2a67eep-6f,
       0x1.84fbbep-9f},
      // [4,5), center=4.5, max_rel_err=2.22e-07 (chebyfit)
      {0x1.3a140ap+1f, 0x1.638d48p+0f, 0x1.fd62f6p-4f, -0x1.52194ep-7f,
       0x1.4db306p-10f},
      // [5,6), center=5.5, max_rel_err=5.49e-08 (chebyfit)
      {0x1.fa99a6p+1f, 0x1.9c70b4p+0f, 0x1.98409cp-4f, -0x1.b23a9cp-8f,
       0x1.5870b8p-11f},
      // [6,7), center=6.5, max_rel_err=5.71e-08 (chebyfit)
      {0x1.6a676ap+2f, 0x1.cafc4ap+0f, 0x1.548ce6p-4f, -0x1.2e1bc6p-8f,
       0x1.906b32p-12f},
      // [7,8), center=7.5, max_rel_err=1.04e-08 (chebyfit)
      {0x1.e23306p+2f, 0x1.f25ebap+0f, 0x1.2413c4p-4f, -0x1.bc6a74p-9f,
       0x1.f9ac9ep-13f},
  };

  if (LIBC_UNLIKELY(x == 1.0 || x == 2.0))
    return 0.0;

  if (x >= 8.0) {
    // Stirling series; 0.5*ln(2*pi)
    //   > mpmath.mp.dps = 50; hex(0.5 * mpmath.log(2 * mpmath.pi))
    constexpr double HALF_LN_2PI = 0x1.d67f1c864beb5p-1;
    double lx = math::log(x);
    double x2 = x * x;
    double result = (x - 0.5) * lx - x + HALF_LN_2PI;
    result += 1.0 / (12.0 * x) - 1.0 / (360.0 * x * x2);
    return result;
  }

  // For x in (0, 4): apply recurrence relation
  // lgamma(x) = lgamma(x+n) - ln(x*(x+1)*...*(x+n-1))
  // to shift x into the stable [4, 8) range for polynomial evaluation.
  // x already in [4, 8) needs no shift -- evaluate the polynomial directly.
  double log_product, xs, product;

  if (x >= 4.0) {
    log_product = 0.0;
    xs = x;
  } else if (x >= 3.0) {
    log_product = math::log(x);
    xs = x + 1.0;
  } else if (x >= 2.0) {
    product = x * (x + 1.0);
    log_product = math::log(product);
    xs = x + 2.0;
  } else if (x >= 1.0) {
    product = x * (x + 1.0);
    product = product * (x + 2.0);
    log_product = math::log(product);
    xs = x + 3.0;
  } else {
    product = x * (x + 1.0);
    product = product * (x + 2.0);
    product = product * (x + 3.0);
    log_product = math::log(product);
    xs = x + 4.0;
  }

  // xs in [4, 8); select polynomial interval.
  float xf = static_cast<float>(xs);
  int n = static_cast<int>(xf);
  if (n >= 7)
    n = 7;
  float t = xf - (static_cast<float>(n) + 0.5f);

  float c0 = LGAMMA_POLY[n - 1][0];
  float c1 = LGAMMA_POLY[n - 1][1];
  float c2 = LGAMMA_POLY[n - 1][2];
  float c3 = LGAMMA_POLY[n - 1][3];
  float c4 = LGAMMA_POLY[n - 1][4];

  // Estrin's scheme for p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4:
  //   p(t) = (c0 + c1*t) + t^2 * ((c2 + c3*t) + t^2 * c4)
  float t2 = t * t;
  float p01 = fputil::multiply_add(t, c1, c0);
  float p23 = fputil::multiply_add(t, c3, c2);
  float p234 = fputil::multiply_add(t2, c4, p23);
  float lgamma_xs_f = fputil::multiply_add(t2, p234, p01);
  double lgamma_xs = static_cast<double>(lgamma_xs_f);

  return lgamma_xs - log_product;
}
} // namespace lgammabf16_internal

LIBC_INLINE bfloat16 lgammabf16(bfloat16 x) {
  using FPBits = fputil::FPBits<bfloat16>;
  FPBits x_bits(x);

  // Handles NaN
  if (LIBC_UNLIKELY(x_bits.is_nan())) {
    if (x_bits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  uint16_t x_u = x_bits.uintval();
  uint16_t x_abs = x_u & 0x7fffU;

  // +Inf or -Inf -> +Inf
  if (LIBC_UNLIKELY(x_abs == 0x7f80U))
    return FPBits::inf(Sign::POS).get_val();

  // +-0 -> +Inf (pole error)
  if (LIBC_UNLIKELY(x_abs == 0U)) {
    fputil::set_errno_if_required(ERANGE);
    fputil::raise_except_if_required(FE_DIVBYZERO);
    return FPBits::inf(Sign::POS).get_val();
  }

  float xf = static_cast<float>(x);

  // Negative integers -> +Inf (pole error)
  if (LIBC_UNLIKELY(x_bits.is_neg())) {
    int biased_exp = x_abs >> FPBits::FRACTION_LEN;
    if (biased_exp >= FPBits::EXP_BIAS) {
      int e = biased_exp - FPBits::EXP_BIAS;
      if (e >= FPBits::FRACTION_LEN ||
          (x_bits.get_mantissa() &
           static_cast<uint16_t>((1U << (FPBits::FRACTION_LEN - e)) - 1U)) ==
              0U) {
        fputil::set_errno_if_required(ERANGE);
        fputil::raise_except_if_required(FE_DIVBYZERO);
        return FPBits::inf(Sign::POS).get_val();
      }
    }

    // Negative non-integer: reflection formula
    // lgamma(x) = ln(pi) - ln|sin(pi*x)| - lgamma(1-x)
    //   > mpmath.mp.dps = 50; hex(mpmath.log(mpmath.pi))
    constexpr double LN_PI_D = 0x1.250d048e7a1bdp+0;
    float ax = -xf;

    // nearest_integer avoids truncation-toward-zero of static_cast<int>
    float frac = fputil::abs(ax - fputil::nearest_integer(ax));
    if (frac > 0.5f)
      frac = 1.0f - frac;

    // sin(pi*frac) via degree-4 Taylor series in double.
    //   > mpmath.mp.dps = 50; hex(mpmath.pi)
    constexpr double PI_D = 0x1.921fb54442d18p+1;
    double frac_d = static_cast<double>(frac);
    double x_pi_d = PI_D * frac_d;
    double x_pi2_d = x_pi_d * x_pi_d;

    // Taylor coefficients for sin(y)/y = 1 - y^2/3! + y^4/5! - y^6/7! + y^8/9!,
    // evaluated at u = y^2 (so q(u) = 1 + DC[0]*u + DC[1]*u^2 + DC[2]*u^3 +
    // DC[3]*u^4). DC[k] = (-1)^(k+1) / (2k+3)! for k = 0..3, generated via:
    //   > mpmath.mp.dps = 50
    //   > [hex((-1)**(k+1) * mpmath.mpf(1) / mpmath.factorial(2*k+3)) for k in
    //   range(4)]
    constexpr double DC[4] = {
        -0x1.5555555555555p-3,  // -1/6       = -1/3!
        0x1.1111111111111p-7,   //  1/120      =  1/5!
        -0x1.a01a01a01a01ap-13, // -1/5040     = -1/7!
        0x1.71de3a556c734p-19,  //  1/362880   =  1/9!
    };

    // Estrin's scheme for q(u) = 1 + DC[0]*u + DC[1]*u^2 + DC[2]*u^3 +
    // DC[3]*u^4 where u = x_pi2_d:
    //   q(u) = (1 + DC[0]*u) + u^2 * ((DC[1] + DC[2]*u) + u^2 * DC[3])
    double u2 = x_pi2_d * x_pi2_d;
    double q01 = fputil::multiply_add(x_pi2_d, DC[0], 1.0);
    double q12 = fputil::multiply_add(x_pi2_d, DC[2], DC[1]);
    double q123 = fputil::multiply_add(u2, DC[3], q12);
    double poly = fputil::multiply_add(u2, q123, q01);
    double sin_pi_frac_d = x_pi_d * poly;

    // A fast (not correctly-rounded) log suffices here: the final result is
    // cast to bfloat16, so we only need ~8 bits of accuracy in log_sin_d.
    double log_sin_d = (sin_pi_frac_d == 1.0)
                           ? 0.0
                           : math::log(static_cast<float>(sin_pi_frac_d));

    // Use double addition: 1.0 + double(ax) preserves tiny ax values that
    // would be lost by 1.0f + ax in float (e.g. ax=2e-5 rounds to 1.0f).
    double lgp_d =
        lgammabf16_internal::lgamma_positive_d(1.0 + static_cast<double>(ax));
    double result_d = LN_PI_D - log_sin_d - lgp_d;

    // Cast directly from double->bfloat16 (not via float) to avoid
    // double-rounding at bfloat16 tie points.
    return fputil::cast<bfloat16>(result_d);
  }

  // Cast directly from double->bfloat16 for the same reason as above.
  return fputil::cast<bfloat16>(
      lgammabf16_internal::lgamma_positive_d(static_cast<double>(xf)));
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_LGAMMABF16_H

//===-- Single-precision atan function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atanf.h"
#include "inv_trigf_utils.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/except_value_utils.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float, atanf, (float x)) {
  using FPBits = typename fputil::FPBits<float>;

  constexpr double FINAL_SIGN[2] = {1.0, -1.0};
  constexpr double SIGNED_PI_OVER_2[2] = {0x1.921fb54442d18p0,
                                          -0x1.921fb54442d18p0};

  FPBits x_bits(x);
  Sign sign = x_bits.sign();
  x_bits.set_sign(Sign::POS);
  uint32_t x_abs = x_bits.uintval();

  // x is inf or nan, |x| < 2^-4 or |x|= > 16.
  if (LIBC_UNLIKELY(x_abs <= 0x3d80'0000U || x_abs >= 0x4180'0000U)) {
    double x_d = static_cast<double>(x);
    double const_term = 0.0;
    if (LIBC_UNLIKELY(x_abs >= 0x4180'0000)) {
      // atan(+-Inf) = +-pi/2.
      if (x_bits.is_inf()) {
        volatile double sign_pi_over_2 = SIGNED_PI_OVER_2[sign.is_neg()];
        return static_cast<float>(sign_pi_over_2);
      }
      if (x_bits.is_nan())
        return x;
      // x >= 16
      x_d = -1.0 / x_d;
      const_term = SIGNED_PI_OVER_2[sign.is_neg()];
    }
    // 0 <= x < 1/16;
    if (LIBC_UNLIKELY(x_bits.is_zero()))
      return x;
    // x <= 2^-12;
    if (LIBC_UNLIKELY(x_abs < 0x3980'0000)) {
#if defined(LIBC_TARGET_CPU_HAS_FMA)
      return fputil::multiply_add(x, -0x1.0p-25f, x);
#else
      double x_d = static_cast<double>(x);
      return static_cast<float>(fputil::multiply_add(x_d, -0x1.0p-25, x_d));
#endif // LIBC_TARGET_CPU_HAS_FMA
    }
    // Use Taylor polynomial:
    //   atan(x) ~ x * (1 - x^2 / 3 + x^4 / 5 - x^6 / 7 + x^8 / 9 - x^10 / 11).
    double x2 = x_d * x_d;
    double x4 = x2 * x2;
    double c0 = fputil::multiply_add(x2, ATAN_COEFFS[0][1], ATAN_COEFFS[0][0]);
    double c1 = fputil::multiply_add(x2, ATAN_COEFFS[0][3], ATAN_COEFFS[0][2]);
    double c2 = fputil::multiply_add(x2, ATAN_COEFFS[0][5], ATAN_COEFFS[0][4]);
    double p = fputil::polyeval(x4, c0, c1, c2);
    double r = fputil::multiply_add(x_d, p, const_term);
    return static_cast<float>(r);
  }

  // Range reduction steps:
  // 1)  atan(x) = sign(x) * atan(|x|)
  // 2)  If |x| > 1, atan(|x|) = pi/2 - atan(1/|x|)
  // 3)  For 1/16 < x <= 1, we find k such that: |x - k/16| <= 1/32.
  // 4)  Then we use polynomial approximation:
  //   atan(x) ~ atan((k/16) + (x - (k/16)) * Q(x - k/16)
  //           = P(x - k/16)
  double x_d, const_term, final_sign;
  int idx;

  if (x_abs > 0x3f80'0000U) {
    // Exceptional value:
    if (LIBC_UNLIKELY(x_abs == 0x3ffe'2ec1U)) { // |x| = 0x1.fc5d82p+0
      return sign.is_pos() ? fputil::round_result_slightly_up(0x1.1ab2fp0f)
                           : fputil::round_result_slightly_down(-0x1.1ab2fp0f);
    }
    // |x| > 1, we need to invert x, so we will perform range reduction in
    // double precision.
    x_d = 1.0 / static_cast<double>(x_bits.get_val());
    double k_d = fputil::nearest_integer(x_d * 0x1.0p4);
    x_d = fputil::multiply_add(k_d, -0x1.0p-4, x_d);
    idx = static_cast<int>(k_d);
    final_sign = FINAL_SIGN[sign.is_pos()];
    // Adjust constant term of the polynomial by +- pi/2.
    const_term = fputil::multiply_add(final_sign, ATAN_COEFFS[idx][0],
                                      SIGNED_PI_OVER_2[sign.is_neg()]);
  } else {
    // Exceptional value:
    if (LIBC_UNLIKELY(x_abs == 0x3dbb'6ac7U)) { // |x| = 0x1.76d58ep-4
      return sign.is_pos()
                 ? fputil::round_result_slightly_up(0x1.75cb06p-4f)
                 : fputil::round_result_slightly_down(-0x1.75cb06p-4f);
    }
    // Perform range reduction in single precision.
    float x_f = x_bits.get_val();
    float k_f = fputil::nearest_integer(x_f * 0x1.0p4f);
    x_f = fputil::multiply_add(k_f, -0x1.0p-4f, x_f);
    x_d = static_cast<double>(x_f);
    idx = static_cast<int>(k_f);
    final_sign = FINAL_SIGN[sign.is_neg()];
    const_term = final_sign * ATAN_COEFFS[idx][0];
  }

  double p = atan_eval(x_d, idx);
  double r = fputil::multiply_add(final_sign * x_d, p, const_term);

  return static_cast<float>(r);
}

} // namespace LIBC_NAMESPACE

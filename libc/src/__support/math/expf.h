//===-- Implementation header for expf --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_H

#if !(defined(LIBC_MATH_HAS_SMALL_TABLES) &&                                   \
      defined(LIBC_MATH_HAS_INTERMEDIATE_COMP_IN_FLOAT))
#include "exp_float_constants.h" // Lookup tables EXP_M1 and EXP_M2.
#endif
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE float expf(float x) {
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);

  uint32_t x_u = xbits.uintval();
  uint32_t x_abs = x_u & 0x7fff'ffffU;

#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
  // Exceptional values
  if (LIBC_UNLIKELY(x_u == 0xc236'bd8cU)) { // x = -0x1.6d7b18p+5f
    return 0x1.108a58p-66f - x * 0x1.0p-95f;
  }
#endif // !LIBC_MATH_HAS_SKIP_ACCURATE_PASS

  // When |x| >= 89, |x| < 2^-25, or x is nan
  if (LIBC_UNLIKELY(x_abs >= 0x42b2'0000U || x_abs <= 0x3280'0000U)) {
    // |x| < 2^-25
    if (xbits.get_biased_exponent() <= 101) {
      return 1.0f + x;
    }

    // When x < log(2^-150) or nan
    if (xbits.uintval() >= 0xc2cf'f1b5U) {
      // exp(-Inf) = 0
      if (xbits.is_inf())
        return 0.0f;
      // exp(nan) = nan
      if (xbits.is_nan())
        return x;
#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
      if (fputil::fenv_is_round_up())
        return FPBits::min_subnormal().get_val();
#endif
      fputil::set_errno_if_required(ERANGE);
      fputil::raise_except_if_required(FE_UNDERFLOW);
      return 0.0f;
    }
    // x >= 89 or nan
    if (xbits.is_pos() && (xbits.uintval() >= 0x42b2'0000)) {
      // x is finite
      if (xbits.uintval() < 0x7f80'0000U) {
#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
        int rounding = fputil::quick_get_round();
        if (rounding == FE_DOWNWARD || rounding == FE_TOWARDZERO)
          return FPBits::max_normal().get_val();
#endif

        fputil::set_errno_if_required(ERANGE);
        fputil::raise_except_if_required(FE_OVERFLOW);
      }
      // x is +inf or nan
      return x + FPBits::inf().get_val();
    }
  }

#if defined(LIBC_MATH_HAS_SMALL_TABLES) &&                                     \
    defined(LIBC_MATH_HAS_INTERMEDIATE_COMP_IN_FLOAT)
  // Range reduction with N=8: x = k * (ln(2)/8) + r, |r| <= ln(2)/16
  constexpr float EIGHT_OVER_LN2 = 0x1.715476p+3f;
  constexpr float LN2_OVER_8_HI = 0x1.62e400p-4f;
  constexpr float LN2_OVER_8_LO = 0x1.7f7d1cp-23f;

  // 2^(j/8) split into high and low floats for j = 0..7 (64 bytes total)
  constexpr float EXP2_HI[8] = {
      0x1.000000p+0f, 0x1.172b84p+0f, 0x1.306fe0p+0f, 0x1.4bfdaep+0f,
      0x1.6a09e6p+0f, 0x1.8ace54p+0f, 0x1.ae89fap+0f, 0x1.d5818ep+0f,
  };
  constexpr float EXP2_LO[8] = {
      0x0.000000p+0f,  -0x1.1ee04ap-25f, 0x1.46275cp-25f,  -0x1.a95e4cp-25f,
      0x1.9fcef2p-25f, 0x1.1d73b2p-25f,  -0x1.b916bep-25f, -0x1.0cc922p-25f,
  };

  float kf = fputil::nearest_integer(x * EIGHT_OVER_LN2);
  int k = static_cast<int>(kf);
  float r = fputil::multiply_add(kf, -LN2_OVER_8_HI, x);
  r = fputil::multiply_add(kf, -LN2_OVER_8_LO, r);

  int idx = k & 7;
  int exp_k = k >> 3;

  float m_hi = EXP2_HI[idx];
  float m_lo = EXP2_LO[idx];

  // exp(r) - 1 ~ r * (1 + r/2 + r^2/6 + r^3/24)
  float p =
      fputil::polyeval(r, 1.0f, 0x1.000000p-1f, 0x1.555556p-3f, 0x1.555556p-5f);
  float pr = r * p;
  float poly = fputil::multiply_add(m_hi, pr, m_lo) + m_hi;

  if (LIBC_UNLIKELY(exp_k <= -126 || exp_k >= 128)) {
    if (exp_k >= 128) {
      if (x > 0x1.62e42ep+6f) {
        fputil::set_errno_if_required(ERANGE);
        fputil::raise_except_if_required(FE_OVERFLOW);
        return FPBits::inf().get_val();
      }
      return (poly * 0x1.0p64f) * 0x1.0p64f;
    }
    float scale1 = 0x1.0p-100f;
    float scale2 =
        FPBits::create_value(Sign::POS, (exp_k + 100) + FPBits::EXP_BIAS, 0)
            .get_val();
    return (poly * scale1) * scale2;
  }
  float scale =
      FPBits::create_value(Sign::POS, exp_k + FPBits::EXP_BIAS, 0).get_val();
  return poly * scale;
#else
  // For -104 < x < 89, to compute exp(x), we perform the following range
  // reduction: find hi, mid, lo such that:
  //   x = hi + mid + lo, in which
  //     hi is an integer,
  //     mid * 2^7 is an integer
  //     -2^(-8) <= lo < 2^-8.
  // In particular,
  //   hi + mid = round(x * 2^7) * 2^(-7).
  // Then,
  //   exp(x) = exp(hi + mid + lo) = exp(hi) * exp(mid) * exp(lo).
  // We store exp(hi) and exp(mid) in the lookup tables EXP_M1 and EXP_M2
  // respectively.  exp(lo) is computed using a degree-4 minimax polynomial
  // generated by Sollya.

  // x_hi = (hi + mid) * 2^7 = round(x * 2^7).
  float kf = fputil::nearest_integer(x * 0x1.0p7f);
  // Subtract (hi + mid) from x to get lo.
  double xd = static_cast<double>(fputil::multiply_add(kf, -0x1.0p-7f, x));
  int x_hi = static_cast<int>(kf);
  x_hi += 104 << 7;
  // hi = x_hi >> 7
  double exp_hi = EXP_M1[x_hi >> 7];
  // mid * 2^7 = x_hi & 0x0000'007fU;
  double exp_mid = EXP_M2[x_hi & 0x7f];
  // Degree-4 minimax polynomial generated by Sollya with the following
  // commands:
  //   > display = hexadecimal;
  //   > Q = fpminimax(expm1(x)/x, 3, [|D...|], [-2^-8, 2^-8]);
  //   > Q;
  double exp_lo =
      fputil::polyeval(xd, 0x1p0, 0x1.ffffffffff777p-1, 0x1.000000000071cp-1,
                       0x1.555566668e5e7p-3, 0x1.55555555ef243p-5);
  return static_cast<float>(exp_hi * exp_mid * exp_lo);
#endif
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXPF_H

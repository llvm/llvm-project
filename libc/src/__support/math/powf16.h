//===-- Implementation header for powf16 ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_POWF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_POWF16_H
#include "include/llvm-libc-macros/float16-macros.h"

#ifdef LIBC_TYPES_HAS_FLOAT16

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/macros/properties/types.h"
#include "src/__support/math/common_constants.h"
#include "src/__support/math/exp10f_utils.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

namespace powf16_impl {
// TODO: mark as constexpr when nearest_integer issue is resolved
LIBC_INLINE static double exp2_range_reduced(double x) {
  // k = round(x * 32)  => (hi + mid) * 2^5
  double kf = fputil::nearest_integer(x * 32.0);
  int k = static_cast<int>(kf);
  // dx = lo = x - (hi + mid) = x - k * 2^(-5)
  double dx = fputil::multiply_add(-0x1.0p-5, kf, x); // -2^-5 * k + x

  // hi = k >> MID_BITS
  // exp_hi = hi shifted into double exponent field
  int64_t hi = static_cast<int64_t>(k >> ExpBase::MID_BITS);
  int64_t exp_hi = static_cast<int64_t>(
      static_cast<uint64_t>(hi) << fputil::FPBits<double>::FRACTION_LEN);

  // mh_bits = bits for 2^hi * 2^mid  (lookup contains base bits for 2^mid)
  int tab_index = k & ExpBase::MID_MASK; // mid index in [0, 31]
  int64_t mh_bits = ExpBase::EXP_2_MID[tab_index] + exp_hi;

  // mh = 2^(hi + mid)
  double mh = fputil::FPBits<double>(static_cast<uint64_t>(mh_bits)).get_val();

  // Degree-5 polynomial approximating (2^x - 1)/x generating by Sollya with:
  // > P = fpminimax((2^x - 1)/x, 5, [|D...|], [-1/32. 1/32]);
  constexpr double COEFFS[5] = {0x1.62e42fefa39efp-1, 0x1.ebfbdff8131c4p-3,
                                0x1.c6b08d7061695p-5, 0x1.3b2b1bee74b2ap-7,
                                0x1.5d88091198529p-10};

  double dx_sq = dx * dx;
  double c1 = fputil::multiply_add(dx, COEFFS[0], 1.0); // 1 + ln2*dx
  double c2 =
      fputil::multiply_add(dx, COEFFS[2], COEFFS[1]); // COEFF1 + COEFF2*dx
  double c3 =
      fputil::multiply_add(dx, COEFFS[4], COEFFS[3]); // COEFF3 + COEFF4*dx
  double p = fputil::multiply_add(dx_sq, c3, c2);     // c2 + c3*dx^2

  // 2^x = 2^(hi+mid) * 2^dx
  //     ≈ mh * (1 + dx * P(dx))
  //     = mh + (mh * dx) * P(dx)
  double result = fputil::multiply_add(p, dx_sq * mh, c1 * mh);

  return result;
}

LIBC_INLINE constexpr bool is_odd_integer(float16 x) {
  using FPBits = fputil::FPBits<float16>;
  FPBits xbits(x);
  uint16_t x_u = xbits.uintval();
  unsigned x_e = static_cast<unsigned>(xbits.get_biased_exponent());
  unsigned lsb = static_cast<unsigned>(
      cpp::countr_zero(static_cast<uint32_t>(x_u | FPBits::EXP_MASK)));
  constexpr unsigned UNIT_EXPONENT =
      static_cast<unsigned>(FPBits::EXP_BIAS + FPBits::FRACTION_LEN);
  return (x_e + lsb == UNIT_EXPONENT);
}

LIBC_INLINE constexpr bool is_integer(float16 x) {
  using FPBits = fputil::FPBits<float16>;
  FPBits xbits(x);
  uint16_t x_u = xbits.uintval();
  unsigned x_e = static_cast<unsigned>(xbits.get_biased_exponent());
  unsigned lsb = static_cast<unsigned>(
      cpp::countr_zero(static_cast<uint32_t>(x_u | FPBits::EXP_MASK)));
  constexpr unsigned UNIT_EXPONENT =
      static_cast<unsigned>(FPBits::EXP_BIAS + FPBits::FRACTION_LEN);
  return (x_e + lsb >= UNIT_EXPONENT);
}

} // namespace powf16_impl

LIBC_INLINE static constexpr float16 powf16(float16 x, float16 y) {
  using namespace powf16_impl;
  using namespace common_constants_internal;
  using FPBits = fputil::FPBits<float16>;

  FPBits xbits(x), ybits(y);
  bool x_sign = xbits.is_neg();
  bool y_sign = ybits.is_neg();

  FPBits x_abs = xbits.abs();
  FPBits y_abs = ybits.abs();

  uint16_t x_u = xbits.uintval();
  uint16_t x_a = x_abs.uintval();
  uint16_t y_a = y_abs.uintval();
  uint16_t y_u = ybits.uintval();
  bool result_sign = false;

  ///////// BEGIN - Check exceptional cases ////////////////////////////////////
  // If x or y is signaling NaN
  if (xbits.is_signaling_nan() || ybits.is_signaling_nan()) {
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  if (LIBC_UNLIKELY(
          ybits.is_zero() || x_u == FPBits::one().uintval() || xbits.is_nan() ||
          ybits.is_nan() || x_u == FPBits::one().uintval() ||
          x_u == FPBits::zero().uintval() || x_u >= FPBits::inf().uintval() ||
          y_u >= FPBits::inf().uintval() ||
          x_u < FPBits::min_normal().uintval() || y_a == 0x3400U || // 0.25
          y_a == 0x3800U ||                                         // 0.5
          y_a == 0x3A00U ||                                         // 0.75
          y_a == 0x3D00U ||                                         // 1.25
          y_a == 0x3E00U ||                                         // 1.5
          y_a == 0x4000U ||                                         // 2.0
          y_a == 0x4100U ||                                         // 2.5
          y_a == 0x4300U ||                                         // 3.5
          is_integer(y))) {
    // pow(x, 0) = 1
    if (ybits.is_zero()) {
      return 1.0f16;
    }

    // pow(1, Y) = 1
    if (x_u == FPBits::one().uintval()) {
      return 1.0f16;
    }
    // 4. Handle remaining NaNs
    // pow(NaN, y) = NaN (for y != 0)
    if (xbits.is_nan()) {
      return x;
    }
    // pow(x, NaN) = NaN (for x != 1)
    if (ybits.is_nan()) {
      return y;
    }
    switch (y_a) {
    case 0x3400U: // y = ±0.25 (1/4)
    case 0x3800U: // y = ±0.5 (1/2)
    case 0x3A00U: // y = ±0.75 (3/4)
    case 0x3D00U: // y = ±1.25 (5/4)
    case 0x3E00U: // y = ±1.5 (3/2)
    case 0x4100U: // y = ±2.5 (5/2)
    case 0x4300U: // y = ±3.5 (7/2)
    {
      if (xbits.is_zero()) {
        if (y_sign) {
          // pow(±0, negative) handled below
          break;
        } else {
          // pow(±0, positive_fractional) = +0
          return FPBits::zero(Sign::POS).get_val();
        }
      }

      if (x_sign && !xbits.is_zero()) {
        break; // pow(negative, non-integer) = NaN
      }

      double x_d = static_cast<double>(x);
      double sqrt_x = fputil::sqrt<double>(x_d);
      double fourth_root = fputil::sqrt<double>(sqrt_x);
      double result_d = 0.0;

      // Compute based on exponent value
      switch (y_a) {
      case 0x3400U: // 0.25 = x^(1/4)
        result_d = fourth_root;
        break;
      case 0x3800U: // 0.5 = x^(1/2)
        result_d = sqrt_x;
        break;
      case 0x3A00U: // 0.75 = x^(1/2) * x^(1/4)
        result_d = sqrt_x * fourth_root;
        break;
      case 0x3D00U: // 1.25 = x * x^(1/4)
        result_d = x_d * fourth_root;
        break;
      case 0x3E00U: // 1.5 = x * x^(1/2)
        result_d = x_d * sqrt_x;
        break;
      case 0x4100U: // 2.5 = x^2 * x^(1/2)
        result_d = x_d * x_d * sqrt_x;
        break;
      case 0x4300U: // 3.5 = x^3 * x^(1/2)
        result_d = x_d * x_d * x_d * sqrt_x;
        break;
      }

      result_d = y_sign ? (1.0 / result_d) : result_d;
      return fputil::cast<float16>(result_d);
    }
    case 0x3c00U: // y = +-1.0
      return fputil::cast<float16>(y_sign ? (1.0 / x) : x);

    case 0x4000U: // y = +-2.0
      double result_d = static_cast<double>(x) * static_cast<double>(x);
      return fputil::cast<float16>(y_sign ? (1.0 / (result_d)) : (result_d));
    }
    // TODO: Speed things up with pow(2, y) = exp2(y) and pow(10, y) = exp10(y).
    //
    // pow(-1, y) for integer y
    if (x_u == FPBits::one(Sign::NEG).uintval()) {
      if (is_integer(y)) {
        if (is_odd_integer(y)) {
          return -1.0f16;
        } else {
          return 1.0f16;
        }
      }
      // pow(-1, non-integer) = NaN
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }

    // pow(±0, y) cases
    if (xbits.is_zero()) {
      if (y_sign) {
        // pow(+-0, negative) = +-inf and raise FE_DIVBYZERO
        fputil::raise_except_if_required(FE_DIVBYZERO);
        bool result_neg = x_sign && ybits.is_finite() && is_odd_integer(y);
        return FPBits::inf(result_neg ? Sign::NEG : Sign::POS).get_val();
      } else {
        // pow(+-0, positive) = +-0
        bool out_is_neg = x_sign && is_odd_integer(y);
        return out_is_neg ? FPBits::zero(Sign::NEG).get_val()
                          : FPBits::zero(Sign::POS).get_val();
      }
    }

    if (xbits.is_inf()) {
      bool out_is_neg = x_sign && ybits.is_finite() && is_odd_integer(y);
      if (y_sign) // pow(+-inf, negative) = +-0
        return out_is_neg ? FPBits::zero(Sign::NEG).get_val()
                          : FPBits::zero(Sign::POS).get_val();
      // pow(+-inf, positive) = +-inf
      return FPBits::inf(out_is_neg ? Sign::NEG : Sign::POS).get_val();
    }

    // y = +-inf cases
    if (ybits.is_inf()) {
      // pow(1, inf) handled above.
      bool x_abs_less_than_one = x_a < FPBits::one().uintval();
      if ((x_abs_less_than_one && !y_sign) ||
          (!x_abs_less_than_one && y_sign)) {
        // |x| < 1 and y = +inf => 0.0
        // |x| > 1 and y = -inf => 0.0
        return 0.0f16;
      } else {
        // |x| > 1 and y = +inf => +inf
        // |x| < 1 and y = -inf => +inf
        return FPBits::inf(Sign::POS).get_val();
      }
    }

    // pow( negative, non-integer ) = NaN
    if (x_sign && !is_integer(y)) {
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }

    bool result_sign = false;
    if (x_sign && is_integer(y)) {
      result_sign = is_odd_integer(y);
    }

    if (is_integer(y)) {
      double base = x_abs.get_val();
      double res = 1.0;
      int yi = static_cast<int>(y_abs.get_val());

      // Fast exponentiation by squaring
      while (yi > 0) {
        if (yi & 1)
          res *= base;
        base *= base;
        yi = yi >> 1;
      }

      if (y_sign) {
        res = 1.0 / res;
      }

      if (result_sign) {
        res = -res;
      }

      if (FPBits(fputil::cast<float16>(res)).is_inf()) {
        fputil::raise_except_if_required(FE_OVERFLOW);
        res = result_sign ? -0x1.0p20 : 0x1.0p20;
      }

      float16 final_res = fputil::cast<float16>(res);
      return final_res;
    }
  }

  ///////// END - Check exceptional cases //////////////////////////////////////

  // Core computation: x^y = 2^( y * log2(x) )
  // We compute log2(x) = log(x) / log(2) using a polynomial approximation.

  // The exponent part (m) is added later to get the final log(x).
  FPBits x_bits(x);
  uint16_t x_u_log = x_bits.uintval();

  // Extract exponent field of x.
  int m = x_bits.get_exponent();

  // When x is subnormal, normalize it by adjusting m.
  if ((x_u_log & FPBits::EXP_MASK) == 0U) {
    unsigned leading_zeros =
        cpp::countl_zero(static_cast<uint32_t>(x_u_log)) - (32 - 16);

    constexpr unsigned SUBNORMAL_SHIFT_CORRECTION = 5;
    unsigned shift = leading_zeros - SUBNORMAL_SHIFT_CORRECTION;

    x_bits.set_mantissa(static_cast<uint16_t>(x_u_log << shift));

    m = 1 - FPBits::EXP_BIAS - static_cast<int>(shift);
  }

  // Extract the mantissa and index into small lookup tables.
  uint16_t mant = x_bits.get_mantissa();
  // Use the highest 7 fractional bits of the mantissa as the index f.
  int f = mant >> (FPBits::FRACTION_LEN - 7);

  // Reconstruct the mantissa value m_x so it's in the range [1.0, 2.0).
  x_bits.set_biased_exponent(FPBits::EXP_BIAS);
  double mant_d = x_bits.get_val();
  // Degree-5 polynomial approximation
  // of log2 generated by Sollya with:
  // > P = fpminimax(log2(1 + x)/x, 4, [|1, D...|], [-2^-8, 2^-7]);
  constexpr double COEFFS[5] = {0x1.71547652b8133p0, -0x1.71547652d1e33p-1,
                                0x1.ec70a098473dep-2, -0x1.7154c5ccdf121p-2,
                                0x1.2514fd90a130ap-2};

#ifdef LIBC_TARGET_CPU_HAS_FMA_DOUBLE
  double v = fputil::multiply_add<double>(mant_d, RD[f], -1.0);
#else
  double c = fputil::FPBits<double>(fputil::FPBits<double>(mant_d).uintval() &
                                    0x3fff'e000'0000'0000)
                 .get_val();
  double v = fputil::multiply_add(RD[f], mant_d - c, CD[f]);
#endif // LIBC_TARGET_CPU_HAS_FMA_DOUBLE
  double extra_factor = static_cast<double>(m) + LOG2_R[f];
  double vsq = v * v;
  double c0 = fputil::multiply_add(v, COEFFS[0], 0.0);
  double c1 = fputil::multiply_add(v, COEFFS[2], COEFFS[1]);
  double c2 = fputil::multiply_add(v, COEFFS[4], COEFFS[3]);

  double log2_x = fputil::polyeval(vsq, c0, c1, c2);

  double y_d = fputil::cast<double>(y);
  double z = fputil::multiply_add(y_d, log2_x, y_d * extra_factor);

  // Check for underflow
  // Float16 min normal is 2^-14, smallest subnormal is 2^-24
  if (LIBC_UNLIKELY(z < -25.0)) {
    fputil::raise_except_if_required(FE_UNDERFLOW);
    return result_sign ? FPBits::zero(Sign::NEG).get_val()
                       : FPBits::zero(Sign::POS).get_val();
  }

  // Check for overflow
  // Float16 max is ~2^16
  double result_d = 0.0;
  if (LIBC_UNLIKELY(z > 16.0)) {
    fputil::raise_except_if_required(FE_OVERFLOW);
    result_d = result_sign ? -0x1.0p20 : 0x1.0p20;
  } else {
    result_d = exp2_range_reduced(z);
  }

  if (result_sign) {

    result_d = -result_d;
  }

  float16 result = fputil::cast<float16>((result_d));
  return result;
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TYPES_HAS_FLOAT16

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXPF16_H

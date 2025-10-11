//===-- Half-precision x^y function ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/powf16.h"
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
#include "src/math/generic/common_constants.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

bool is_odd_integer(float16 x) {
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

bool is_integer(float16 x) {
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

} // namespace

LLVM_LIBC_FUNCTION(float16, powf16, (float16 x, float16 y)) {
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
          ybits.is_zero() || x_u == FPBits::one().uintval() ||
          x_u == FPBits::one().uintval() || x_u == FPBits::zero().uintval() ||
          x_u >= FPBits::inf().uintval() || y_u >= FPBits::inf().uintval() ||
          x_u < FPBits::min_normal().uintval() || y_a == 0x3800U ||
          y_a == 0x3c00U || y_a == 0x4000U || is_integer(y))) {
    // pow(x, 0) = 1
    if (ybits.is_zero()) {
      return fputil::cast<float16>(1.0f);
    }

    // pow(1, Y) = 1
    if (x_u == FPBits::one().uintval()) {
      return fputil::cast<float16>(1.0f);
    }

    switch (y_a) {

    case 0x3800U: { // y = +-0.5
      if (LIBC_UNLIKELY(
              (x == 0.0 || x_u == FPBits::inf(Sign::NEG).uintval()))) {
        // pow(-0, 1/2) = +0
        // pow(-inf, 1/2) = +inf
        // For pow(x, 0.5), sqrt(x) is used. pow(0, -0.5) is handled below.
        break;
      }
      // If x is not negative or special, use sqrt(x)
      if (x_sign && !xbits.is_zero()) {
        // pow(negative, non-integer) = NaN, handled below.
        break;
      }
      return fputil::cast<float16>(y_sign ? (1.0 / fputil::sqrt<float16>(x))
                                          : fputil::sqrt<float16>(x));
    }
    case 0x3c00U: // y = +-1.0
      return fputil::cast<float16>(y_sign ? (1.0 / x) : x);

    case 0x4000U: // y = +-2.0
      return fputil::cast<float16>(y_sign ? (1.0 / (x * x)) : (x * x));
    }
    // TODO: Speed things up with pow(2, y) = exp2(y) and pow(10, y) = exp10(y).

    // Propagate remaining quiet NaNs.
    if (xbits.is_quiet_nan()) {
      return x;
    }
    if (ybits.is_quiet_nan()) {
      return y;
    }

    // x = -1: special case for integer exponents
    if (x_u == FPBits::one(Sign::NEG).uintval()) {
      if (is_integer(y)) {
        if (is_odd_integer(y)) {
          return fputil::cast<float16>(-1.0f);
        } else {
          return fputil::cast<float16>(1.0f);
        }
      }
      // pow(-1, non-integer) = NaN
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }

    // x = 0 cases
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
        return fputil::cast<float16>(0.0f);
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
    if (is_integer(y)) {
      double base = x_abs.get_val();
      double res = 1.0;
      int yi = static_cast<int>(y_abs.get_val());

      while (yi > 0) {
        if (yi % 2 == 1)
          res *= base;
        base *= base;
        yi /= 2;
      }

      if (y_sign) {
        res = 1.0 / res;
      }

      float16 final_res = fputil::cast<float16>(res);

      if (x_sign && is_odd_integer(y)) {
        FPBits res_bits(final_res);
        res_bits.set_sign(Sign::NEG);
        return res_bits.get_val();
      }
      return final_res;
    }
    // For negative x with integer y, compute pow(|x|, y) and adjust sign
    if (x_sign) {
      x = -x;
      if (is_odd_integer(y)) {
        result_sign = true;
      }
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

  // Range reduction for log(m_x):
  // v = r * m_x - 1

  // log(1+v) = v + v^2 * P(v)

  // log(m_x) = log(1+v) - log(r).

  double log_m_x;

  double v = fputil::multiply_add<double>(mant_d, R[f], -1.0);

  double p_v = fputil::polyeval(v, LOG_COEFFS[0], LOG_COEFFS[1], LOG_COEFFS[2],
                                LOG_COEFFS[3], LOG_COEFFS[4], LOG_COEFFS[5]);

  // log(1+v) = v + v^2 * P(v)
  double logp1_v = fputil::multiply_add<double>(v * v, p_v, v);

  // log(m_x) = log(1+v) - log(r).
  log_m_x = logp1_v + LOG_R[f];

  // Complete log(x) = m * log(2) + log(m_x).
  double log_x =
      fputil::multiply_add<double>(static_cast<double>(m), LOG_2_HI, log_m_x);
  // Convert to log2(x): log2(x) = log(x) / log(2)
  double log2_x = log_x / LOG_2_HI;

  // z = y * log2(x)
  // Now compute 2^z = 2^(n + r), with n integer and r in [-0.5, 0.5].
  double z = fputil::cast<double>(y) * log2_x;

  // Check for underflow. Half-precision min normal exponent is -14 ,
  // but the smallest subnormal is 2^-24.
  if (LIBC_UNLIKELY(z < -25.0)) {
    fputil::raise_except_if_required(FE_UNDERFLOW);
    // Determine sign of underflowed result
    return result_sign ? FPBits::zero(Sign::NEG).get_val()
                       : FPBits::zero(Sign::POS).get_val();
  }

  // Check for overflow. The max value of float16 is ~2^16, so z > 16.0 will
  // overflow.
  if (LIBC_UNLIKELY(z > 16.0)) {
    fputil::raise_except_if_required(FE_OVERFLOW);

    float16 max_finite = FPBits::max_normal().get_val();
    float16 neg_max_finite = FPBits::max_normal(Sign::NEG).get_val();

    int round_mode = fputil::get_round();

    if (result_sign) {
      // Negative result overflows
      // For TOWARDZERO or DOWNWARD rounding, the result is the most negative
      // finite value.
      if (round_mode == FE_TOWARDZERO || round_mode == FE_DOWNWARD) {
        return neg_max_finite;
      }
      // For other rounding modes, the result is -Infinity.
      return FPBits::inf(Sign::NEG).get_val();
    } else {
      // Positive result overflows
      // For TOWARDZERO or DOWNWARD rounding, the result is the max finite
      // value.
      if (round_mode == FE_TOWARDZERO || round_mode == FE_DOWNWARD) {
        return max_finite;
      }
      // For other rounding modes, the result is +Infinity.
      return FPBits::inf(Sign::POS).get_val();
    }
  }

  double n = fputil::nearest_integer(z);
  double r = z - n;

  // Compute 2^r using a degree-7 polynomial for r in [-0.5, 0.5].
  // Generated by Sollya with:
  // > P = fpminimax(2^x, 7, [|D...|], [-0.5, 0.5]);
  // The polynomial coefficients are rounded from the Sollya output.
  constexpr double EXP2_COEFFS[] = {
      0x1p+0,                // 1.0
      0x1.62e42fefa39efp-1,  // ln(2)
      0x1.ebfbdff82c58fp-3,  // ln(2)^2 / 2
      0x1.c6b08d704a0c0p-5,  // ln(2)^3 / 6
      0x1.3b2ab6fba4e77p-7,  // ln(2)^4 / 24
      0x1.5d87fe78a6737p-10, // ln(2)^5 / 120
      0x1.430912f86a805p-13, // ln(2)^6 / 720
      0x1.10e4104ac8015p-17  // ln(2)^7 / 5040
  };

  double exp2_r = fputil::polyeval(
      r, EXP2_COEFFS[0], EXP2_COEFFS[1], EXP2_COEFFS[2], EXP2_COEFFS[3],
      EXP2_COEFFS[4], EXP2_COEFFS[5], EXP2_COEFFS[6], EXP2_COEFFS[7]);

  // Compute 2^n by direct bit manipulation.
  int n_int = static_cast<int>(n);
  uint64_t exp_bits = static_cast<uint64_t>(n_int + 1023) << 52;
  double pow2_n = cpp::bit_cast<double>(exp_bits);
  float16 result = fputil::cast<float16>((pow2_n * exp2_r));

  if (result_sign) {
    FPBits result_bits(result);
    result_bits.set_sign(Sign::NEG);
    result = result_bits.get_val();
  }

  return result;
}

} // namespace LIBC_NAMESPACE_DECL

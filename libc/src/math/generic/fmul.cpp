//===-- Implementation of fmul function------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmul.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/common.h"
#include "src/__support/uint128.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float, fmul, (double x, double y)) {
  auto x_bits = fputil::FPBits<double>(x);

  auto y_bits = fputil::FPBits<double>(y);

  auto output_sign = (x_bits.sign() != y_bits.sign()) ? Sign::NEG : Sign::POS;

  if (LIBC_UNLIKELY(x_bits.is_inf_or_nan() || y_bits.is_inf_or_nan() ||
                    x_bits.is_zero() || y_bits.is_zero())) {
    if (x_bits.is_nan())
      return static_cast<float>(x);
    if (y_bits.is_nan())
      return static_cast<float>(y);
    if (x_bits.is_inf())
      return y_bits.is_zero()
                 ? fputil::FPBits<float>::quiet_nan().get_val()
                 : fputil::FPBits<float>::inf(output_sign).get_val();
    if (y_bits.is_inf())
      return x_bits.is_zero()
                 ? fputil::FPBits<float>::quiet_nan().get_val()
                 : fputil::FPBits<float>::inf(output_sign).get_val();
    // Now either x or y is zero, and the other one is finite.
    return fputil::FPBits<float>::zero(output_sign).get_val();
  }

  uint64_t mx, my;

  // Get mantissa and append the hidden bit if needed.
  mx = x_bits.get_explicit_mantissa();
  my = y_bits.get_explicit_mantissa();

  // Get the corresponding biased exponent.
  int ex = x_bits.get_explicit_exponent();
  int ey = y_bits.get_explicit_exponent();

  // Count the number of leading zeros of the explicit mantissas.
  int nx = cpp::countl_zero(mx);
  int ny = cpp::countl_zero(my);
  // Shift the leading 1 bit to the most significant bit.
  mx <<= nx;
  my <<= ny;

  // Adjust exponent accordingly: If x or y are normal, we will only need to
  // shift by (exponent length + sign bit = 11 bits. If x or y are denormal, we
  // will need to shift more than 11 bits.
  ex -= (nx - 11);
  ey -= (ny - 11);

  UInt128 product = static_cast<UInt128>(mx) * static_cast<UInt128>(my);
  int32_t dm1;
  uint64_t highs, lows;
  uint64_t g, hight, lowt;
  uint32_t m;
  uint32_t b;
  int c;

  highs = static_cast<uint64_t>(product >> 64);
  c = static_cast<int>(highs >= 0x8000000000000000);
  lows = static_cast<uint64_t>(product);

  lowt = (lows != 0);

  dm1 = ex + ey + c + fputil::FPBits<float>::EXP_BIAS;

  int round_mode = fputil::quick_get_round();
  if (dm1 >= 255) {
    if ((round_mode == FE_TOWARDZERO) ||
        (round_mode == FE_UPWARD && output_sign.is_neg()) ||
        (round_mode == FE_DOWNWARD && output_sign.is_pos())) {
      return fputil::FPBits<float>::max_normal(output_sign).get_val();
    }
    return fputil::FPBits<float>::inf().get_val();
  } else if (dm1 <= 0) {

    int m_shift = 40 + c - dm1;
    int g_shift = m_shift - 1;
    int h_shift = 64 - g_shift;
    m = (m_shift >= 64) ? 0 : static_cast<uint32_t>(highs >> m_shift);

    g = g_shift >= 64 ? 0 : (highs >> g_shift) & 1;
    hight = h_shift >= 64 ? highs : (highs << h_shift) != 0;

    dm1 = 0;
  } else {
    m = static_cast<uint32_t>(highs >> (39 + c));
    g = (highs >> (38 + c)) & 1;
    hight = (highs << (26 - c)) != 0;
  }

  if (round_mode == FE_TONEAREST) {
    b = g && ((hight && lowt) || ((m & 1) != 0));
  } else if ((output_sign.is_neg() && round_mode == FE_DOWNWARD) ||
             (output_sign.is_pos() && round_mode == FE_UPWARD)) {
    b = (g == 0 && (hight && lowt) == 0) ? 0 : 1;
  } else {
    b = 0;
  }

  uint32_t exp16 = (dm1 << 23);

  uint32_t m2 = m & fputil::FPBits<float>::FRACTION_MASK;

  uint32_t result = (exp16 + m2) + b;

  auto result_bits = fputil::FPBits<float>(result);
  result_bits.set_sign(output_sign);
  return result_bits.get_val();
}

} // namespace LIBC_NAMESPACE

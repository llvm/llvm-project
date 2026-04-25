//===-- Implementation header for rsqrtf16 ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_RSQRTF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_RSQRTF16_H

#include "include/llvm-libc-macros/float16-macros.h"

#ifdef LIBC_TYPES_HAS_FLOAT16

#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

namespace rsqrtf16_internal {

LIBC_INLINE_VAR constexpr int RSQRT_FRACTION_BITS = 29;
LIBC_INLINE_VAR constexpr int64_t ONE = int64_t(1) << RSQRT_FRACTION_BITS;
LIBC_INLINE_VAR constexpr int64_t THREE_HALVES = 3 * (ONE >> 1);

// Degree-4 minimax polynomial generated with Sollya:
//   P = fpminimax(1/sqrt(x), 4,
//       [|single,single,single,single,single|], [0.5;1])
// Coefficients are stored in Q29 fixed-point format.
LIBC_INLINE_VAR constexpr int64_t COEFFS[5] = {
    1'573'164'416, -2'940'085'504, 3'653'406'208, -2'366'894'080, 617'319'616,
};
LIBC_INLINE_VAR constexpr int64_t ONE_OVER_SQRT2 = 0x16a09e60;

LIBC_INLINE constexpr int floor_log2(uint64_t x) {
  int result = -1;
  while (x) {
    x >>= 1;
    ++result;
  }
  return result;
}

LIBC_INLINE constexpr int64_t eval_polynomial(uint32_t m) {
  int64_t y = COEFFS[4];
  y = COEFFS[3] + ((y * m) >> RSQRT_FRACTION_BITS);
  y = COEFFS[2] + ((y * m) >> RSQRT_FRACTION_BITS);
  y = COEFFS[1] + ((y * m) >> RSQRT_FRACTION_BITS);
  y = COEFFS[0] + ((y * m) >> RSQRT_FRACTION_BITS);
  return y;
}

LIBC_INLINE constexpr int64_t newton_raphson(uint32_t m, int64_t y) {
  int64_t y2 = (y * y) >> RSQRT_FRACTION_BITS;
  int64_t my2 = (static_cast<int64_t>(m) * y2) >> RSQRT_FRACTION_BITS;
  int64_t factor = THREE_HALVES - (my2 >> 1);
  return (y * factor) >> RSQRT_FRACTION_BITS;
}

LIBC_INLINE constexpr uint16_t fixed_to_half_bits(uint64_t y, int scale_exp) {
  int y_log2 = floor_log2(y);
  int out_exp = scale_exp + y_log2 - RSQRT_FRACTION_BITS;
  int biased_exp = out_exp + 15;

  uint32_t out_sig = y_log2 >= 10 ? static_cast<uint32_t>(y >> (y_log2 - 10))
                                  : static_cast<uint32_t>(y << (10 - y_log2));

  if (biased_exp <= 0)
    return 0x0400;
  if (biased_exp >= 31)
    return 0x7bff;

  return static_cast<uint16_t>((biased_exp << 10) | (out_sig & 0x3ff));
}

LIBC_INLINE constexpr uint16_t approximate_rsqrt(uint16_t x_abs) {
  uint32_t x_mant = x_abs & 0x03ff;
  int exponent = 0;

  if (x_abs >= 0x0400) {
    x_mant |= 0x0400;
    exponent = static_cast<int>(x_abs >> 10) - 14;
  } else {
    exponent = -13;
    while ((x_mant & 0x0400) == 0) {
      x_mant <<= 1;
      --exponent;
    }
  }

  uint32_t m = x_mant << (RSQRT_FRACTION_BITS - 11);
  int64_t y = newton_raphson(m, eval_polynomial(m));

  int scale_exp = 0;
  if (exponent & 1) {
    y = (y * ONE_OVER_SQRT2) >> RSQRT_FRACTION_BITS;
    scale_exp = -((exponent - 1) / 2);
  } else {
    scale_exp = -(exponent / 2);
  }

  return fixed_to_half_bits(static_cast<uint64_t>(y), scale_exp);
}

// Compare y = sig * 2^exp with 1 / sqrt(x_sig * 2^x_exp).
// Return -1 if y is below the exact value, 0 if exact, and 1 if above.
LIBC_INLINE constexpr int compare_with_rsqrt(uint32_t sig, int exp,
                                             uint32_t x_sig, int x_exp) {
  uint64_t lhs = static_cast<uint64_t>(sig) * sig * x_sig;
  int scale = 2 * exp + x_exp;

  if (scale >= 0)
    return (scale == 0 && lhs == 1) ? 0 : 1;

  int rshift = -scale;
  if (rshift >= 64)
    return -1;

  uint64_t rhs = uint64_t(1) << rshift;
  if (lhs < rhs)
    return -1;
  if (lhs > rhs)
    return 1;
  return 0;
}

LIBC_INLINE constexpr int compare_half_with_rsqrt(uint16_t y, uint32_t x_sig,
                                                  int x_exp) {
  uint32_t y_sig = 0x0400 | (y & 0x03ff);
  int y_exp = static_cast<int>(y >> 10) - 25;
  return compare_with_rsqrt(y_sig, y_exp, x_sig, x_exp);
}

LIBC_INLINE constexpr uint16_t floor_rsqrt(uint16_t approx, uint32_t x_sig,
                                           int x_exp) {
  uint16_t y = approx < 0x0400 ? 0x0400 : approx;
  while (compare_half_with_rsqrt(y, x_sig, x_exp) > 0)
    --y;
  while (y < 0x7bff && compare_half_with_rsqrt(y + 1, x_sig, x_exp) <= 0)
    ++y;
  return y;
}

LIBC_INLINE constexpr uint16_t round_result(uint16_t y, uint32_t x_sig,
                                            int x_exp) {
  if (compare_half_with_rsqrt(y, x_sig, x_exp) == 0)
    return y;

  int rounding_mode = FE_TONEAREST;
  if (!cpp::is_constant_evaluated())
    rounding_mode = fputil::get_round();
  if (rounding_mode == FE_UPWARD)
    return y + 1;
  if (rounding_mode != FE_TONEAREST)
    return y;

  uint32_t y_sig = 0x0400 | (y & 0x03ff);
  int y_exp = static_cast<int>(y >> 10) - 25;
  uint32_t midpoint_sig = (y_sig << 1) | 1;
  int midpoint_cmp = compare_with_rsqrt(midpoint_sig, y_exp - 1, x_sig, x_exp);

  if (midpoint_cmp < 0)
    return y + 1;
  if (midpoint_cmp > 0)
    return y;
  return (y & 1) ? static_cast<uint16_t>(y + 1) : y;
}

LIBC_INLINE constexpr float16 rsqrtf16_no_float(uint16_t x_abs) {
  uint32_t x_sig = 0;
  int x_exp = 0;
  if (x_abs >= 0x0400) {
    x_sig = 0x0400 | (x_abs & 0x03ff);
    x_exp = static_cast<int>(x_abs >> 10) - 25;
  } else {
    x_sig = x_abs;
    x_exp = -24;
  }

  uint16_t approx = approximate_rsqrt(x_abs);
  uint16_t y = floor_rsqrt(approx, x_sig, x_exp);
  return fputil::FPBits<float16>(round_result(y, x_sig, x_exp)).get_val();
}

} // namespace rsqrtf16_internal

LIBC_INLINE constexpr float16 rsqrtf16(float16 x) {
  using FPBits = fputil::FPBits<float16>;
  FPBits xbits(x);

  uint16_t x_u = xbits.uintval();
  uint16_t x_abs = x_u & 0x7fff;

  constexpr uint16_t INF_BIT = FPBits::inf().uintval();

  // x is 0, inf/nan, or negative.
  if (LIBC_UNLIKELY(x_u == 0 || x_u >= INF_BIT)) {
    // x is NaN
    if (x_abs > INF_BIT) {
      if (xbits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return x;
    }

    // |x| = 0
    if (x_abs == 0) {
      fputil::raise_except_if_required(FE_DIVBYZERO);
      fputil::set_errno_if_required(ERANGE);
      return FPBits::inf(xbits.sign()).get_val();
    }

    // -inf <= x < 0
    if (x_u > 0x7fff) {
      fputil::raise_except_if_required(FE_INVALID);
      fputil::set_errno_if_required(EDOM);
      return FPBits::quiet_nan().get_val();
    }

    // x = +inf => rsqrt(x) = +0
    return FPBits::zero(xbits.sign()).get_val();
  }

#ifdef LIBC_TARGET_CPU_HAS_FPU_FLOAT
  float result = 1.0f / fputil::sqrt<float>(fputil::cast<float>(x));

  // Targeted post-corrections to ensure correct rounding in half for specific
  // mantissa patterns
  const uint16_t half_mantissa = x_abs & 0x3ff;
  if (LIBC_UNLIKELY(half_mantissa == 0x011F)) {
    result = fputil::multiply_add(result, 0x1.0p-21f, result);
  } else if (LIBC_UNLIKELY(half_mantissa == 0x0313)) {
    result = fputil::multiply_add(result, -0x1.0p-21f, result);
  }

  return fputil::cast<float16>(result);

#else
  return rsqrtf16_internal::rsqrtf16_no_float(x_abs);
#endif
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TYPES_HAS_FLOAT16

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_RSQRTF16_H

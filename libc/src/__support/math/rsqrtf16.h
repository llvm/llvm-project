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
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

static constexpr float16 rsqrtf16(float16 x) {
  using FPBits = fputil::FPBits<float16>;
  FPBits xbits(x);

  uint16_t x_u = xbits.uintval();
  uint16_t x_abs = x_u & 0x7fff;
  uint16_t x_sign = x_u >> 15;

  // x is NaN
  if (LIBC_UNLIKELY(xbits.is_nan())) {
    if (xbits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // |x| = 0
  if (LIBC_UNLIKELY(x_abs == 0x0)) {
    fputil::raise_except_if_required(FE_DIVBYZERO);
    fputil::set_errno_if_required(ERANGE);
    return FPBits::inf(Sign::POS).get_val();
  }

  // -inf <= x < 0
  if (LIBC_UNLIKELY(x_sign == 1)) {
    fputil::raise_except_if_required(FE_INVALID);
    fputil::set_errno_if_required(EDOM);
    return FPBits::quiet_nan().get_val();
  }

  // x = +inf => rsqrt(x) = 0
  if (LIBC_UNLIKELY(xbits.is_inf())) {
    return fputil::cast<float16>(0.0f);
  }

  // x is valid, estimate the result
  // Range reduction:
  // x can be expressed as m*2^e, where e - int exponent and m - mantissa
  // rsqrtf16(x) = rsqrtf16(m*2^e)
  // rsqrtf16(m*2^e) = 1/sqrt(m) * 1/sqrt(2^e) = 1/sqrt(m) * 1/2^(e/2)
  // 1/sqrt(m) * 1/2^(e/2) = 1/sqrt(m) * 2^(-e/2)

  // Compute in float throughout to minimize cost while preserving accuracy.
  float xf = x;
  int exponent = 0;
  float mantissa = fputil::frexp(xf, exponent);

  float result = 0.0f;
  int exp_floored = -(exponent >> 1);

  if (mantissa == 0.5f) {
    // When mantissa is 0.5f, x was a power of 2 (or subnormal that normalizes
    // this way). 1/sqrt(0.5f) = sqrt(2.0f).
    // If exponent is odd (exponent = 2k + 1):
    //   rsqrt(x) = (1/sqrt(0.5)) * 2^(-(2k+1)/2) = sqrt(2) * 2^(-k-0.5)
    //            = sqrt(2) * 2^(-k) * (1/sqrt(2)) = 2^(-k)
    //   exp_floored = -((2k+1)>>1) = -(k) = -k
    //   So result = ldexp(1.0f, exp_floored)
    // If exponent is even (exponent = 2k):
    //   rsqrt(x) = (1/sqrt(0.5)) * 2^(-2k/2) = sqrt(2) * 2^(-k)
    //   exp_floored = -((2k)>>1) = -(k) = -k
    //   So result = ldexp(sqrt(2.0f), exp_floored)
    if (exponent & 1) {
      result = fputil::ldexp(1.0f, exp_floored);
    } else {
      constexpr float SQRT_2_F = 0x1.6a09e6p0f; // sqrt(2.0f)
      result = fputil::ldexp(SQRT_2_F, exp_floored);
    }
  } else {
    // Degree-5 polynomial (float coefficients) generated with Sollya:
    // P = fpminimax(1/sqrt(x) + 2^-28, 5, [|single...|], [0.5,1])
    float y =
        fputil::polyeval(mantissa, 0x1.9c81fap1f, -0x1.e2c63ap2f, 0x1.91e9b8p3f,
                         -0x1.899abep3f, 0x1.9eddeap2f, -0x1.6bdb48p0f);

    // Newton-Raphson iteration in float (use multiply_add to leverage FMA when
    // available):
    float y2 = y * y;
    float factor = fputil::multiply_add(-0.5f * mantissa, y2, 1.5f);
    y = y * factor;

    result = fputil::ldexp(y, exp_floored);
    if (exponent & 1) {
      constexpr float ONE_OVER_SQRT2 = 0x1.6a09e6p-1f; // 1/sqrt(2)
      result *= ONE_OVER_SQRT2;
    }

    // Targeted post-correction: for the specific half-precision mantissa
    // pattern M == 0x011F we observe a consistent -1 ULP bias across exponents.
    // Apply a tiny upward nudge to cross the rounding boundary in all modes.
    const uint16_t half_mantissa = static_cast<uint16_t>(x_abs & 0x3ff);
    if (half_mantissa == 0x011F) {
      // Nudge up to fix consistent -1 ULP at that mantissa boundary
      result = fputil::multiply_add(result, 0x1.0p-21f,
                                    result); // result *= (1 + 2^-21)
    } else if (half_mantissa == 0x0313) {
      // Nudge down to fix +1 ULP under upward rounding at this mantissa
      // boundary
      result = fputil::multiply_add(result, -0x1.0p-21f,
                                    result); // result *= (1 - 2^-21)
    }
  }

  return fputil::cast<float16>(result);
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TYPES_HAS_FLOAT16

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_RSQRTF16_H

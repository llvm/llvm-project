//===-- Half-precision rsqrt function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "src/math/rsqrtf16.h"
#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FMA.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, rsqrtf16, (float16 x)) {
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

  float xf = x;
  int exponent;
  float mantissa = fputil::frexp(xf, exponent);

  // 6-degree polynomial generated using Sollya
  // bigger polynomial doesn't generate better results-> the current one
  // produces the least number of errors but still errors are presents P =
  // fpminimax(1/(sqrt(x)), [|0,1,2,3,4,5|], [|SG...|], [0.5, 1]);
  float interm =
      fputil::polyeval(mantissa, 0x1.9c81c4p1f, -0x1.e2c57cp2f, 0x1.91e8bp3f,
                       -0x1.899954p3f, 0x1.9edcp2f, -0x1.6bd93cp0f);

  // Apply one Newton-Raphson iteration to refine the approximation of
  // 1/sqrt(mantissa) y_new = y_old * (1.5 - 0.5 * mantissa * y_old^2) Using
  // fputil::fma for potential precision benefits in the factor calculation
  float interm_sq = interm * interm;
  float factor = fputil::fma<float>(-0.5f * mantissa, interm_sq, 1.5f);
  float interm_refined = interm * factor; // Final multiplication

  // Apply a second Newton-Raphson iteration
  // y_new = y_old * (1.5 - 0.5 * mantissa * y_old^2)
  // y_old is now interm_refined
  float interm_refined_sq = interm_refined * interm_refined;
  float factor2 = fputil::fma<float>(-0.5f * mantissa, interm_refined_sq, 1.5f);
  float interm_refined2 = interm_refined * factor2;

  // Round (-e/2)
  int exp_floored = -(exponent >> 1);

  // rsqrt(x) = 1/sqrt(mantissa) * 2^(-e/2)
  // rsqrt(x) = P(mantissa) * 2*(exp_floored)
  // float result = fputil::ldexp(interm, exp_floored);
  float result = fputil::ldexp(interm_refined2, exp_floored);

  // Handle the case where exponent is odd
  if (exponent & 1) {
    const float ONE_OVER_SQRT2 = 0x1.6a09e6p-1f;
    // result *= ONE_OVER_SQRT2;
    result = fputil::fma<float>(result, ONE_OVER_SQRT2,
                                0.0f); // Use FMA for multiplication
  }

  return fputil::cast<float16>(result);
}
} // namespace LIBC_NAMESPACE_DECL

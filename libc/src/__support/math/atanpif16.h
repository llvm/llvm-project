//===-- Implementation header for atanpif16 ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ATANHIF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ATANHIF16_H

#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

LIBC_INLINE constexpr float16 atanpif16(float16 x) {
  using FPBits = fputil::FPBits<float16>;

  FPBits xbits(x);
  bool is_neg = xbits.is_neg();

  auto signed_result = [is_neg](double r) -> float16 {
    return fputil::cast<float16>(is_neg ? -r : r);
  };

  if (LIBC_UNLIKELY(xbits.is_inf_or_nan())) {
    if (xbits.is_nan()) {
      if (xbits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return x;
    }
    // atanpi(±∞) = ±0.5
    return signed_result(0.5);
  }

  if (LIBC_UNLIKELY(xbits.is_zero()))
    return x;

  double x_abs = fputil::cast<double>(xbits.abs().get_val());

  if (LIBC_UNLIKELY(x_abs == 1.0))
    return signed_result(0.25);

  // evaluate atan(x)/pi using polynomial approximation, valid for |x| <= 0.5
  constexpr auto atanpi_eval = [](double x) -> double {
    // polynomial coefficients for atan(x)/pi taylor series
    // generated using sympy: series(atan(x)/pi, x, 0, 17)
    constexpr static double POLY_COEFFS[] = {
        0x1.45f306dc9c889p-2,  // x^1:   1/pi
        -0x1.b2995e7b7b60bp-4, // x^3:  -1/(3*pi)
        0x1.04c26be3b06ccp-4,  // x^5:   1/(5*pi)
        -0x1.7483758e69c08p-5, // x^7:  -1/(7*pi)
        0x1.21bb945252403p-5,  // x^9:   1/(9*pi)
        -0x1.da1bace3cc68ep-6, // x^11: -1/(11*pi)
        0x1.912b1c2336cf2p-6,  // x^13:  1/(13*pi)
        -0x1.5bade52f95e7p-6,  // x^15: -1/(15*pi)
    };
    double x_sq = x * x;
    return x * fputil::polyeval(x_sq, POLY_COEFFS[0], POLY_COEFFS[1],
                                POLY_COEFFS[2], POLY_COEFFS[3], POLY_COEFFS[4],
                                POLY_COEFFS[5], POLY_COEFFS[6], POLY_COEFFS[7]);
  };

  // Case 1: |x| <= 0.5 - Direct polynomial evaluation
  if (LIBC_LIKELY(x_abs <= 0.5)) {
    double result = atanpi_eval(x_abs);
    return signed_result(result);
  }

  // case 2: 0.5 < |x| <= 1 - use double-angle reduction
  // atan(x) = 2 * atan(x / (1 + sqrt(1 + x^2)))
  // so atanpi(x) = 2 * atanpi(x') where x' = x / (1 + sqrt(1 + x^2))
  if (x_abs <= 1.0) {
    double x_abs_sq = x_abs * x_abs;
    double sqrt_term = fputil::sqrt<double>(1.0 + x_abs_sq);
    double x_prime = x_abs / (1.0 + sqrt_term);
    double result = 2.0 * atanpi_eval(x_prime);
    return signed_result(result);
  }

  // case 3: |x| > 1 - use reciprocal transformation
  // atan(x) = pi/2 - atan(1/x) for x > 0
  // so atanpi(x) = 1/2 - atanpi(1/x)
  double x_recip = 1.0 / x_abs;
  double result = 0.0;

  // if 1/|x| > 0.5, we need to apply Case 2 transformation to 1/|x|
  if (x_recip > 0.5) {
    double x_recip_sq = x_recip * x_recip;
    double sqrt_term = fputil::sqrt<double>(1.0 + x_recip_sq);
    double x_prime = x_recip / (1.0 + sqrt_term);
    result = fputil::multiply_add(-2.0, atanpi_eval(x_prime), 0.5);
  } else {
    // direct evaluation since 1/|x| <= 0.5
    result = 0.5 - atanpi_eval(x_recip);
  }

  return signed_result(result);
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif

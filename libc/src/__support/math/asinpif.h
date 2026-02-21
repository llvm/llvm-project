//===-- Implementation header for asinpif -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ASINPIF_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ASINPIF_H

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/macros/properties/types.h"

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/except_value_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

LIBC_INLINE constexpr float asinpif(float x) {
#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
  constexpr size_t N_EXCEPTS = 5;
  constexpr fputil::ExceptValues<float, N_EXCEPTS> ASINPIF_EXCEPTS = {
      {// (inputs, RZ output, RU offset, RD offset, RN offset)
       // x = 0x1.e768f6p-122, asinpif(x) = 0x1.364b7ap-123 (RZ)
       {0x02F3B47B, 0x021B25BD, 1, 0, 0},
       // x = 0x1.e768f6p-24, asinpif(x) = 0x1.364b7ap-25 (RZ)
       {0x33F3B47B, 0x331B25BD, 1, 0, 1},
       // x = 0x1.dddb4ep-19, asinpif(x) = 0x1.303686p-20 (RZ)
       {0x366EEDA7, 0x35981B43, 1, 0, 1},
       // x = -0x1.dddb4ep-19, asinpif(x) = -0x1.303686p-20 (RZ)
       {0xB66EEDA7, 0xB5981B43, 0, 1, 1},
       // x = -0x1.e768f6p-24, asinpif(x) = -0x1.364b7ap-25 (RZ)
       {0xB3F3B47B, 0xB31B25BD, 0, 1, 1}}};
#endif // !LIBC_MATH_HAS_SKIP_ACCURATE_PASS

  using FPBits = fputil::FPBits<float>;

  FPBits xbits(x);
  bool is_neg = xbits.is_neg();
  double x_abs = fputil::cast<double>(xbits.abs().get_val());

  auto signed_result = [is_neg](auto r) -> auto { return is_neg ? -r : r; };

  if (LIBC_UNLIKELY(x_abs > 1.0)) {
    if (xbits.is_nan()) {
      if (xbits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return x;
    }

    fputil::raise_except_if_required(FE_INVALID);
    fputil::set_errno_if_required(EDOM);
    return FPBits::quiet_nan().get_val();
  }

#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
  auto r = ASINPIF_EXCEPTS.lookup(xbits.uintval());
  if (LIBC_UNLIKELY(r.has_value()))
    return r.value();
#endif // !LIBC_MATH_HAS_SKIP_ACCURATE_PASS

  // the coefficients for the polynomial approximation of asin(x)/(pi*x) in the
  // range [0, 0.5] extracted using Sollya.
  //
  // Sollya code:
  // > prec = 200;
  // > display = hexadecimal;
  // > g = asin(x) / (pi * x);
  // > P = fpminimax(g, [|0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20|],
  // >              [|D...|], [0, 0.5]);
  // > for i from 0 to degree(P) do coeff(P, i);
  // > print("Error:", dirtyinfnorm(P - g, [1e-30; 0.25]));
  // Error: 0x1.45c281e1cf9b58p-50 ~= 2^−49.652
  //
  // Non-zero coefficients (even powers only):
  constexpr double ASINPI_POLY_COEFFS[] = {
      0x1.45f306dc9c881p-2,  // x^0
      0x1.b2995e7b7e756p-5,  // x^2
      0x1.8723a1d12f828p-6,  // x^4
      0x1.d1a45564b9545p-7,  // x^6
      0x1.3ce4ceaa0e1e9p-7,  // x^8
      0x1.d2c305898ea13p-8,  // x^10
      0x1.692212e27a5f9p-8,  // x^12
      0x1.2b22cc744d25bp-8,  // x^14
      0x1.8427b864479ffp-9,  // x^16
      0x1.815522d7a2bf1p-8,  // x^18
      -0x1.f6df98438aef4p-9, // x^20
      0x1.4b50c2eb13708p-7   // x^22
  };
  // Evaluates P1(v2) = c1 + c2*v2 + c3*v2^2 + ... (tail of P without c0)
  auto asinpi_polyeval = [&](double v2) -> double {
    return fputil::polyeval(
        v2, ASINPI_POLY_COEFFS[1], ASINPI_POLY_COEFFS[2], ASINPI_POLY_COEFFS[3],
        ASINPI_POLY_COEFFS[4], ASINPI_POLY_COEFFS[5], ASINPI_POLY_COEFFS[6],
        ASINPI_POLY_COEFFS[7], ASINPI_POLY_COEFFS[8], ASINPI_POLY_COEFFS[9],
        ASINPI_POLY_COEFFS[10], ASINPI_POLY_COEFFS[11]);
  };

  // if |x| <= 0.5:
  //   asinpi(x) = x * (c0 + x^2 * P1(x^2))
  if (LIBC_UNLIKELY(x_abs <= 0.5)) {
    double x_d = fputil::cast<double>(x);
    double v2 = x_d * x_d;
    double result = x_d * fputil::multiply_add(v2, asinpi_polyeval(v2),
                                               ASINPI_POLY_COEFFS[0]);
    return fputil::cast<float>(result);
  }

  // If |x| > 0.5:
  //   asinpi(x) = 0.5 - 2 * sqrt(u) * P(u)
  //             = 0.5 - 2 * sqrt(u) * (c0 + u * P1(u))
  //             = (0.5 - 2*sqrt(u)*ONE_OVER_PI_HI)
  //               - 2*sqrt(u) * (ONE_OVER_PI_LO + DELTA_C0 + u * P1(u))
  //
  // where u = (1 - |x|) / 2, and
  //   ONE_OVER_PI_HI + ONE_OVER_PI_LO = 1/pi to ~106 bits
  //   DELTA_C0 = c0 - ONE_OVER_PI_HI
  //
  // ONE_OVER_PI_LO + DELTA_C0 is a single precomputed constant:
  //   = ONE_OVER_PI_LO + (c0 - ONE_OVER_PI_HI)
  //   = c0 - (ONE_OVER_PI_HI - ONE_OVER_PI_LO)
  //   = c0 - 1/pi  (to ~106 bits)
  constexpr double ONE_OVER_PI_HI = 0x1.45f306dc9c883p-2;
  constexpr double ONE_OVER_PI_LO = -0x1.6b01ec5417056p-56;
  // C0_MINUS_1OVERPI = c0 - 1/pi = DELTA_C0 + ONE_OVER_PI_LO
  constexpr double C0_MINUS_1OVERPI =
      (ASINPI_POLY_COEFFS[0] - ONE_OVER_PI_HI) + ONE_OVER_PI_LO;

  double u = fputil::multiply_add(-0.5, x_abs, 0.5);
  double sqrt_u = fputil::sqrt<double>(u);
  double neg2_sqrt_u = -2.0 * sqrt_u;

  // tail = (c0 - 1/pi) + u * P1(u)
  double tail = fputil::multiply_add(u, asinpi_polyeval(u), C0_MINUS_1OVERPI);

  double result_hi = fputil::multiply_add(neg2_sqrt_u, ONE_OVER_PI_HI, 0.5);
  double result = fputil::multiply_add(tail, neg2_sqrt_u, result_hi);

  return fputil::cast<float>(signed_result(result));
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ASINPIF_H

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
  // polynomial evaluation using horner's method
  // work only for |x| in [0, 0.5]
  // Returns v * P(v2) where P(v2) = c0 + c1*v2 + c2*v2^2 + ...
  auto asinpi_polyeval = [&](double v, double v2) -> double {
    return v * fputil::polyeval(v2, ASINPI_POLY_COEFFS[0],
                                ASINPI_POLY_COEFFS[1], ASINPI_POLY_COEFFS[2],
                                ASINPI_POLY_COEFFS[3], ASINPI_POLY_COEFFS[4],
                                ASINPI_POLY_COEFFS[5], ASINPI_POLY_COEFFS[6],
                                ASINPI_POLY_COEFFS[7], ASINPI_POLY_COEFFS[8],
                                ASINPI_POLY_COEFFS[9], ASINPI_POLY_COEFFS[10],
                                ASINPI_POLY_COEFFS[11]);
  };

  // Returns P(v2) - c0 = c1*v2 + c2*v2^2 + ...
  // This is the "tail" of the polynomial, used to avoid cancellation
  // in the range reduction path.
  auto asinpi_polyeval_tail = [&](double v2) -> double {
    return v2 * fputil::polyeval(v2, ASINPI_POLY_COEFFS[1],
                                 ASINPI_POLY_COEFFS[2], ASINPI_POLY_COEFFS[3],
                                 ASINPI_POLY_COEFFS[4], ASINPI_POLY_COEFFS[5],
                                 ASINPI_POLY_COEFFS[6], ASINPI_POLY_COEFFS[7],
                                 ASINPI_POLY_COEFFS[8], ASINPI_POLY_COEFFS[9],
                                 ASINPI_POLY_COEFFS[10],
                                 ASINPI_POLY_COEFFS[11]);
  };

  // if |x| <= 0.5:
  if (LIBC_UNLIKELY(x_abs <= 0.5)) {
    double x_d = fputil::cast<double>(x);
    double result = asinpi_polyeval(x_d, x_d * x_d);
    return fputil::cast<float>(result);
  }

  // If |x| > 0.5, we need to use the range reduction method:
  //    y = asin(x) => x = sin(y)
  //      because: sin(a) = cos(pi/2 - a)
  //      therefore:
  //    x = cos(pi/2 - y)
  //      let z = pi/2 - y,
  //    x = cos(z)
  //      because: cos(2a) = 1 - 2 * sin^2(a), z = 2a, a = z/2
  //      therefore:
  //    cos(z) = 1 - 2 * sin^2(z/2)
  //    sin(z/2) = sqrt((1 - cos(z))/2)
  //    sin(z/2) = sqrt((1 - x)/2)
  //      let u = (1 - x)/2
  //      then:
  //    sin(z/2) = sqrt(u)
  //    z/2 = asin(sqrt(u))
  //    z = 2 * asin(sqrt(u))
  //    pi/2 - y = 2 * asin(sqrt(u))
  //    y = pi/2 - 2 * asin(sqrt(u))
  //    y/pi = 1/2 - 2 * asin(sqrt(u))/pi
  //
  // Finally, we can write:
  //   asinpi(x) = 1/2 - 2 * asinpi(sqrt(u))
  //     where u = (1 - x) /2
  //             = 0.5 - 0.5 * x
  //             = multiply_add(-0.5, x, 0.5)
  //
  // asinpi(x) = 0.5 - 2 * sqrt(u) * P(u)
  //
  // To avoid cancellation when |x| is near 0.5 (where 2*sqrt(u)*P(u) ~ 0.5),
  // we split P(u) into its leading term c0 and the tail:
  //   P(u) = c0 + tail(u)
  //
  // We further split the constant 1/pi into high and low parts for precision:
  //   1/pi = ONE_OVER_PI_HI + ONE_OVER_PI_LO
  //
  // And rewrite the expression as:
  //   0.5 - 2*sqrt(u) * (1/pi + (c0 - 1/pi) + tail(u))
  //
  // The term (0.5 - 2*sqrt(u)*ONE_OVER_PI_HI) is computed exactly using FMA.
  // The remaining small terms are added separately:
  //   - 2*sqrt(u) * ONE_OVER_PI_LO
  //   - 2*sqrt(u) * (c0 - 1/pi)      [absorbed into tail sum]
  //   - 2*sqrt(u) * tail(u)

  constexpr double ONE_OVER_PI_HI = 0x1.45f306dc9c883p-2;
  constexpr double ONE_OVER_PI_LO = -0x1.6b01ec5417056p-56;
  // Verify: ONE_OVER_PI_HI + ONE_OVER_PI_LO ≈ 1/pi to ~106 bits

  // DELTA_C0 = c0 - ONE_OVER_PI_HI (difference between Sollya c0 and 1/pi hi)
  constexpr double DELTA_C0 = ASINPI_POLY_COEFFS[0] - ONE_OVER_PI_HI;

  double u = fputil::multiply_add(-0.5, x_abs, 0.5);
  double sqrt_u = fputil::sqrt<double>(u);

  // compute the tail: P(u) - c0
  double tail = asinpi_polyeval_tail(u);

  double neg2_sqrt_u = -2.0 * sqrt_u;
  double result_hi = fputil::multiply_add(neg2_sqrt_u, ONE_OVER_PI_HI, 0.5);
  double result = result_hi + neg2_sqrt_u * (ONE_OVER_PI_LO + DELTA_C0 + tail);

  return fputil::cast<float>(signed_result(result));
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ASINPIF_H

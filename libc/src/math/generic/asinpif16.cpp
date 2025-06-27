//===-- Half-precision asinf16(x) function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/optimization.h"
#include "src/math/asinfpi16.h"

namespace LIBC_NAMESPACE_DECL {

static constexpr float16 ONE_OVER_TWO = 0x3800; // 0.5f16

#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
static constexpr size_t N_ASINFPI_EXCEPTS = 9;

static constexpr float16 ONE_OVER_THREE = 0x3555; // 0.333251953125f16
static constexpr float16 ONE_OVER_FOUR = 0x3400;  // 0.25f16
static constexpr float16 ONE_OVER_SIX = 0x32ab;   // 0.166748046875f16

static constexpr fputil::ExceptValues<float16, N_ASINFPI_EXCEPTS>
    ASINFPI_EXCEPTS{{
        // (input_hex, RZ_output_hex, RU_offset, RD_offset, RN_offset)

        // x = 0.0, asinfpi(0.0) = 0.0
        {0x0000, 0x0000, 0, 0, 0},

        // x = 1.0, asinfpi(1.0) = 0.5
        {0x3C00, ONE_OVER_TWO.uintval(), 0, 0, 0},

        // x = -1.0, asinfpi(-1.0) = -0.5
        {0xBC00, (fputil::FPBits<float16>(-ONE_OVER_TWO)).uintval(), 0, 0, 0},

        // x = 0.5, asinfpi(0.5) = 1/6
        {0x3800, ONE_OVER_SIX.uintval(), 0, 0, 0},

        // x = -0.5, asinfpi(-0.5) = -1/6
        {0xB800, (fputil::FPBits<float16>(-ONE_OVER_SIX)).uintval(), 0, 0, 0},

        // x = sqrt(2)/2 ~ 0.70710678, asinfpi(x) = 1/4
        // 0x3B41 is float16 for ~0.707. 0x3400 is float16 for 0.25
        {0x3B41, ONE_OVER_FOUR.uintval(), 0, 0, 0},

        // x = -sqrt(2)/2 ~ -0.70710678, asinfpi(x) = -1/4
        {0xBB41, (fputil::FPBits<float16>(-ONE_OVER_FOUR)).uintval(), 0, 0, 0},

        // x = sqrt(3)/2 ~ 0.8660254, asinfpi(x) = 1/3
        // 0x3BF2 is float16 for ~0.866. 0x3555 is float16 for 1/3
        {0x3BF2, ONE_OVER_THREE.uintval(), 0, 0, 0},

        // x = -sqrt(3)/2 ~ -0.8660254, asinfpi(x) = -1/3
        {0xBBF2, (fputil::FPBits<float16>(-ONE_OVER_THREE)).uintval(), 0, 0, 0},
    }};
#endif // !LIBC_MATH_HAS_SKIP_ACCURATE_PASS

LLVM_LIBC_FUNCTION(float16, asinpif16, (float16 x)) {
  using FPBits = fputil::FPBits<float16>;

  FPBits xbits(x);
  uint16_t x_uint = xbits.uintval();
  uint16_t x_abs = xbits.uintval() & 0x7fffU;
  uint16_t x_sign = x_uint >> 15;

  if (LIBC_UNLIKELY(x_abs > 0x3c00)) {
    // aspinf16(NaN) = NaN
    if (xbits.is_nan()) {
      if (xbits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return x;
    }

    // 1 < |x| <= +/-inf
    fputil::raise_except_if_required(FE_INVALID);
    fputil::set_errno_if_required(EDOM);

    return FPBits::quiet_nan().get_val();
  }

#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
  // Handle exceptional values
  if (auto r = ACOSF16_EXCEPTS.lookup(x_u); LIBC_UNLIKELY(r.has_value()))
    return r.value();

#else
  // Handling zero
  if (LIBC_UNLIKELY(x_abs == 0x0000)) {
    return x;
  }

  // Handling +/-1.0
  // If x is +/-1.0, return +/-0.5
  if (LIBC_UNLIKELY(x_abs == 0x3c00)) {
    return fputil::cast<float16>(x_sign ? -ONE_OVER_TWO : ONE_OVER_TWO);
  }
#endif // !LIBC_MATH_HAS_SKIP_ACCURATE_PASS

  // the coefficients for the polynomial approximation of asin(x)/pi in the
  // range [0, 0.5] extracted using python-sympy
  //
  // Python code to generate the coefficients:
  //   from sympy import *
  //   import math
  //   x = symbols('x')
  //   print(series(asin(x)/math.pi, x, 0, 21))
  //
  // OUTPUT:
  //
  // 0.318309886183791*x + 0.0530516476972984*x**3 + 0.0238732414637843*x**5 +
  // 0.0142102627760621*x**7 + 0.00967087327815336*x**9 +
  // 0.00712127941391293*x**11 + 0.00552355646848375*x**13 +
  // 0.00444514782463692*x**15 + 0.00367705242846804*x**17 +
  // 0.00310721681820837*x**19 + O(x**21)
  //
  // it's very accurate in the range [0, 0.5] and has a maximum error of
  // 0.0000000000000001 in the range [0, 0.5].
  static constexpr float16 POLY_COEFFS[10] = {
      0.318309886183791f16,   // x^1
      0.0530516476972984f16,  // x^3
      0.0238732414637843f16,  // x^5
      0.0142102627760621f16,  // x^7
      0.00967087327815336f16, // x^9
      0.00712127941391293f16, // x^11
      0.00552355646848375f16, // x^13
      0.00444514782463692f16, // x^15
      0.00367705242846804f16, // x^17
      0.00310721681820837f16  // x^19
  };

  // polynomial evaluation using horner's method
  // work only for |x| in [0, 0.5]
  auto __asinpi_polyeval = [](float16 xsq) -> float16 {
    return fputil::polyeval(xsq, POLY_COEFFS[0], POLY_COEFFS[1], POLY_COEFFS[2],
                            POLY_COEFFS[3], POLY_COEFFS[4], POLY_COEFFS[5],
                            POLY_COEFFS[6], POLY_COEFFS[7], POLY_COEFFS[9],
                            POLY_COEFFS[9]);
  };

  // if |x| <= 0.5:
  if (x_abs <= 0x3800) {
    // Use polynomial approximation of asin(x)/pi in the range [0, 0.5]
    float16 xsq = x * x;
    float16 result = x * __asinpi_polyeval(xsq);
    return fputil::cast<float16>(result);
  }

  // If |x| > 0.5, we need to use the range reduction method:
  //    y = asin(x) => x = sin(y)
  //      because: sin(a) = cos(pi/2 - a)
  //      therefore:
  //    x = cos(pi/2 - y)
  //      let z = pi/2 - y,
  //    x = cos(z)
  //      becuase: cos(2a) = 1 - 2 * sin^2(a), z = 2a, a = z/2
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

  float16 u = fputil::multiply_add(-ONE_OVER_TWO, x, ONE_OVER_TWO);
  float16 asinpi_sqrt_u = __asinpi_polyeval(u);
  float16 result = fputil::multiply_add(-2.0f16, asinpi_sqrt_u, ONE_OVER_TWO);

  return fputil::cast<float16>(result);
}

} // namespace LIBC_NAMESPACE_DECL

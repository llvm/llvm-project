//===-- Implementation header for asinpi ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ASINPI_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ASINPI_H

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/except_value_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

// the coefficients for the polynomial approximation of asin(x)/pi in the
// range [0, 0.5] extracted using python-sympy
//
// Python code to generate the coefficients:
//  > from sympy import *
//  > import math
//  > x = symbols('x')
//  > print(series(asin(x)/math.pi, x, 0, 25))
//
// OUTPUT:
//
// 0.318309886183791*x + 0.0530516476972984*x**3 + 0.0238732414637843*x**5 +
// 0.0142102627760621*x**7 + 0.00967087327815336*x**9 +
// 0.00712127941391293*x**11 + 0.00552355646848375*x**13 +
// 0.00444514782463692*x**15 + 0.00367705242846804*x**17 +
// 0.00310721681820837*x**19 + 0.00267072683660291*x**21 +
// 0.00232764927854127*x**23 + O(x**25)
static constexpr double ASINPI_POLY_COEFFS[] = {
    0x1.45f306dc9c889p-2, // 0.318309886183791 x^1
    0x1.b2995e7b7b5fdp-5, // 0.053051647697298 x^3
    0x1.8723a1d588a36p-6, // 0.023873241463784 x^5
    0x1.d1a452f20430dp-7, // 0.014210262776062 x^7
    0x1.3ce52a3a09f61p-7, // 0.009670873278153 x^9
    0x1.d2b33e303d375p-8, // 0.007121279413913 x^11
    0x1.69fde663c674fp-8, // 0.005523556468484 x^13
    0x1.235134885f19bp-8, // 0.004445147824637 x^15
    0x1.e1f567da15ce2p-9, // 0.003677052428468 x^17
    0x1.9744e53b4851bp-9, // 0.003107216818208 x^19
    0x1.5e0eb8d6eaa8p-9,  // 0.002670726836603 x^21
    0x1.3116f30e47f56p-9, // 0.002327649278541 x^23
};

static constexpr double asinpi(double x) {
  using FPBits = fputil::FPBits<double>;

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

  // polynomial evaluation using horner's method
  // work only for |x| in [0, 0.5]
  auto asinpi_polyeval = [&](double v) -> double {
    return v * fputil::polyeval(v * v, ASINPI_POLY_COEFFS[0],
                                ASINPI_POLY_COEFFS[1], ASINPI_POLY_COEFFS[2],
                                ASINPI_POLY_COEFFS[3], ASINPI_POLY_COEFFS[4],
                                ASINPI_POLY_COEFFS[5], ASINPI_POLY_COEFFS[6],
                                ASINPI_POLY_COEFFS[7], ASINPI_POLY_COEFFS[8],
                                ASINPI_POLY_COEFFS[9], ASINPI_POLY_COEFFS[10],
                                ASINPI_POLY_COEFFS[11]);
  };

  // if |x| <= 0.5:
  if (LIBC_UNLIKELY(x_abs <= 0.5)) {
    return asinpi_polyeval(x);
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

  double u = fputil::multiply_add(-0.5, x_abs, 0.5);
  double asinpi_sqrt_u = asinpi_polyeval(u);
  double result = fputil::multiply_add(-2.0, asinpi_sqrt_u, 0.5);

  return signed_result(result);
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ASINPI_H

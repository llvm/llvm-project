//===-- Compute sin + cos for small angles ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_GENERIC_SINCOS_EVAL_H
#define LLVM_LIBC_SRC_MATH_GENERIC_SINCOS_EVAL_H

#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/multiply_add.h"

namespace LIBC_NAMESPACE {

using fputil::DoubleDouble;

LIBC_INLINE void sincos_eval(const DoubleDouble &u, DoubleDouble &sin_u,
                             DoubleDouble &cos_u) {
  // Evaluate sin(y) = sin(x - k * (pi/128))
  // We use the degree-7 Taylor approximation:
  //   sin(y) ~ y - y^3/3! + y^5/5! - y^7/7!
  // Then the error is bounded by:
  //   |sin(y) - (y - y^3/3! + y^5/5! - y^7/7!)| < |y|^9/9! < 2^-54/9! < 2^-72.
  // For y ~ u_hi + u_lo, fully expanding the polynomial and drop any terms
  // < ulp(u_hi^3) gives us:
  //   y - y^3/3! + y^5/5! - y^7/7! = ...
  // ~ u_hi + u_hi^3 * (-1/6 + u_hi^2 * (1/120 - u_hi^2 * 1/5040)) +
  //        + u_lo (1 + u_hi^2 * (-1/2 + u_hi^2 / 24))
  double u_hi_sq = u.hi * u.hi; // Error < ulp(u_hi^2) < 2^(-6 - 52) = 2^-58.
  // p1 ~ 1/120 + u_hi^2 / 5040.
  double p1 = fputil::multiply_add(u_hi_sq, -0x1.a01a01a01a01ap-13,
                                   0x1.1111111111111p-7);
  // q1 ~ -1/2 + u_hi^2 / 24.
  double q1 = fputil::multiply_add(u_hi_sq, 0x1.5555555555555p-5, -0x1.0p-1);
  double u_hi_3 = u_hi_sq * u.hi;
  // p2 ~ -1/6 + u_hi^2 (1/120 - u_hi^2 * 1/5040)
  double p2 = fputil::multiply_add(u_hi_sq, p1, -0x1.5555555555555p-3);
  // q2 ~ 1 + u_hi^2 (-1/2 + u_hi^2 / 24)
  double q2 = fputil::multiply_add(u_hi_sq, q1, 1.0);
  double sin_lo = fputil::multiply_add(u_hi_3, p2, u.lo * q2);
  // Overall, |sin(y) - (u_hi + sin_lo)| < 2*ulp(u_hi^3) < 2^-69.

  // Evaluate cos(y) = cos(x - k * (pi/128))
  // We use the degree-8 Taylor approximation:
  //   cos(y) ~ 1 - y^2/2 + y^4/4! - y^6/6! + y^8/8!
  // Then the error is bounded by:
  //   |cos(y) - (...)| < |y|^10/10! < 2^-81
  // For y ~ u_hi + u_lo, fully expanding the polynomial and drop any terms
  // < ulp(u_hi^3) gives us:
  //   1 - y^2/2 + y^4/4! - y^6/6! + y^8/8! = ...
  // ~ 1 - u_hi^2/2 + u_hi^4(1/24 + u_hi^2 (-1/720 + u_hi^2/40320)) +
  //     + u_hi u_lo (-1 + u_hi^2/6)
  // We compute 1 - u_hi^2 accurately:
  //   v_hi + v_lo ~ 1 - u_hi^2/2
  double v_hi = fputil::multiply_add(u.hi, u.hi * (-0.5), 1.0);
  double v_lo = 1.0 - v_hi; // Exact
  v_lo = fputil::multiply_add(u.hi, u.hi * (-0.5), v_lo);

  // r1 ~ -1/720 + u_hi^2 / 40320
  double r1 = fputil::multiply_add(u_hi_sq, 0x1.a01a01a01a01ap-16,
                                   -0x1.6c16c16c16c17p-10);
  // s1 ~ -1 + u_hi^2 / 6
  double s1 = fputil::multiply_add(u_hi_sq, 0x1.5555555555555p-3, -1.0);
  double u_hi_4 = u_hi_sq * u_hi_sq;
  double u_hi_u_lo = u.hi * u.lo;
  // r2 ~ 1/24 + u_hi^2 (-1/720 + u_hi^2 / 40320)
  double r2 = fputil::multiply_add(u_hi_sq, r1, 0x1.5555555555555p-5);
  // s2 ~ v_lo + u_hi * u_lo * (-1 + u_hi^2 / 6)
  double s2 = fputil::multiply_add(u_hi_u_lo, s1, v_lo);
  double cos_lo = fputil::multiply_add(u_hi_4, r2, s2);
  // Overall, |cos(y) - (v_hi + cos_lo)| < 2*ulp(u_hi^4) < 2^-75.

  sin_u = fputil::exact_add(u.hi, sin_lo);
  cos_u = fputil::exact_add(v_hi, cos_lo);
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_MATH_GENERIC_SINCOSF_EVAL_H

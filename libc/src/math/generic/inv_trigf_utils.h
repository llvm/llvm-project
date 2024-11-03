//===-- Single-precision general inverse trigonometric functions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_GENERIC_INV_TRIGF_UTILS_H
#define LLVM_LIBC_SRC_MATH_GENERIC_INV_TRIGF_UTILS_H

#include "math_utils.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/__support/common.h"

#include <errno.h>

namespace __llvm_libc {

// PI and PI / 2
constexpr double M_MATH_PI = 0x1.921fb54442d18p+1;
constexpr double M_MATH_PI_2 = 0x1.921fb54442d18p+0;

// atan table size
constexpr int ATAN_T_BITS = 4;
constexpr int ATAN_T_SIZE = 1 << ATAN_T_BITS;

// N[Table[ArcTan[x], {x, 1/8, 8/8, 1/8}], 40]
extern const double ATAN_T[ATAN_T_SIZE];
extern const double ATAN_K[5];

// The main idea of the function is to use formula
// atan(u) + atan(v) = atan((u+v)/(1-uv))

// x should be positive, normal finite value
LIBC_INLINE double atan_eval(double x) {
  using FPB = fputil::FPBits<double>;
  // Added some small value to umin and umax mantissa to avoid possible rounding
  // errors.
  FPB::UIntType umin =
      FPB::create_value(false, FPB::EXPONENT_BIAS - ATAN_T_BITS - 1,
                        0x100000000000UL)
          .uintval();
  FPB::UIntType umax =
      FPB::create_value(false, FPB::EXPONENT_BIAS + ATAN_T_BITS,
                        0xF000000000000UL)
          .uintval();

  FPB bs(x);
  bool sign = bs.get_sign();
  auto x_abs = bs.uintval() & FPB::FloatProp::EXP_MANT_MASK;

  if (x_abs <= umin) {
    double pe = __llvm_libc::fputil::polyeval(x * x, 0.0, ATAN_K[1], ATAN_K[2],
                                              ATAN_K[3], ATAN_K[4]);
    return fputil::multiply_add(pe, x, x);
  }

  if (x_abs >= umax) {
    double one_over_x_m = -1.0 / x;
    double one_over_x2 = one_over_x_m * one_over_x_m;
    double pe = __llvm_libc::fputil::polyeval(one_over_x2, ATAN_K[0], ATAN_K[1],
                                              ATAN_K[2], ATAN_K[3]);
    return fputil::multiply_add(pe, one_over_x_m, sign ? (-M_MATH_PI_2) : (M_MATH_PI_2));
  }

  double pos_x = FPB(x_abs).get_val();
  bool one_over_x = pos_x > 1.0;
  if (one_over_x) {
    pos_x = 1.0 / pos_x;
  }

  double near_x = fputil::nearest_integer(pos_x * ATAN_T_SIZE);
  int val = static_cast<int>(near_x);
  near_x *= 1.0 / ATAN_T_SIZE;

  double v = (pos_x - near_x) / fputil::multiply_add(near_x, pos_x, 1.0);
  double v2 = v * v;
  double pe = __llvm_libc::fputil::polyeval(v2, ATAN_K[0], ATAN_K[1], ATAN_K[2],
                                            ATAN_K[3], ATAN_K[4]);
  double result;
  if (one_over_x)
    result = M_MATH_PI_2 - fputil::multiply_add(pe, v, ATAN_T[val - 1]);
  else
    result = fputil::multiply_add(pe, v, ATAN_T[val - 1]);
  return sign ? -result : result;
}

// > Q = fpminimax(asin(x)/x, [|0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20|],
//                 [|1, D...|], [0, 0.5]);
constexpr double ASIN_COEFFS[10] = {0x1.5555555540fa1p-3, 0x1.333333512edc2p-4,
                                    0x1.6db6cc1541b31p-5, 0x1.f1caff324770ep-6,
                                    0x1.6e43899f5f4f4p-6, 0x1.1f847cf652577p-6,
                                    0x1.9b60f47f87146p-7, 0x1.259e2634c494fp-6,
                                    -0x1.df946fa875ddp-8, 0x1.02311ecf99c28p-5};

// Evaluate P(x^2) - 1, where P(x^2) ~ asin(x)/x
LIBC_INLINE double asin_eval(double xsq) {
  double x4 = xsq * xsq;
  double r1 = fputil::polyeval(x4, ASIN_COEFFS[0], ASIN_COEFFS[2],
                               ASIN_COEFFS[4], ASIN_COEFFS[6], ASIN_COEFFS[8]);
  double r2 = fputil::polyeval(x4, ASIN_COEFFS[1], ASIN_COEFFS[3],
                               ASIN_COEFFS[5], ASIN_COEFFS[7], ASIN_COEFFS[9]);
  return fputil::multiply_add(xsq, r2, r1);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_GENERIC_INV_TRIGF_UTILS_H

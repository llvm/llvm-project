//===-- Single-precision sinh function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sinhf.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/generic/explogxf.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(float, sinhf, (float x)) {
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);
  bool sign = xbits.get_sign();
  uint32_t x_abs = xbits.uintval() & FPBits::FloatProp::EXP_MANT_MASK;

  // |x| <= 2^-26
  if (unlikely(x_abs <= 0x3280'0000U)) {
    return unlikely(x_abs == 0) ? x : (x + 0.25 * x * x * x);
  }

  // When |x| >= 90, or x is inf or nan
  if (unlikely(x_abs >= 0x42b4'0000U)) {
    if (xbits.is_nan())
      return x + 1.0f; // sNaN to qNaN + signal

    if (xbits.is_inf())
      return x;

    int rounding = fputil::get_round();
    if (sign) {
      if (unlikely(rounding == FE_UPWARD || rounding == FE_TOWARDZERO))
        return FPBits(FPBits::MAX_NORMAL | FPBits::FloatProp::SIGN_MASK)
            .get_val();
    } else {
      if (unlikely(rounding == FE_DOWNWARD || rounding == FE_TOWARDZERO))
        return FPBits(FPBits::MAX_NORMAL).get_val();
    }

    errno = ERANGE;

    return x + FPBits::inf(sign).get_val();
  }

  // |x| <= 0.078125
  if (unlikely(x_abs <= 0x3da0'0000U)) {
    // |x| = 0.0005589424981735646724700927734375
    if (unlikely(x_abs == 0x3a12'85ffU)) {
      if (fputil::get_round() == FE_TONEAREST)
        return x;
    }

    double xdbl = x;
    double x2 = xdbl * xdbl;
    // Sollya: fpminimax(sinh(x),[|3,5,7|],[|D...|],[-1/16-1/64;1/16+1/64],x);
    // Sollya output: x * (0x1p0 + x^0x1p1 * (0x1.5555555556583p-3 + x^0x1p1
    //                  * (0x1.111110d239f1fp-7
    //                  + x^0x1p1 * 0x1.a02b5a284013cp-13)))
    // Therefore, output of Sollya = x * pe;
    double pe = fputil::polyeval(x2, 0.0, 0x1.5555555556583p-3,
                                 0x1.111110d239f1fp-7, 0x1.a02b5a284013cp-13);
    return fputil::multiply_add(xdbl, pe, xdbl);
  }

  // sinh(x) = (e^x - e^(-x)) / 2.
  return exp_pm_eval</*is_sinh*/ true>(x);
}

} // namespace __llvm_libc

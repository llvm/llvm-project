//===-- Single-precision atanh function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atanhf.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/math/generic/explogxf.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(float, atanhf, (float x)) {
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);
  bool sign = xbits.get_sign();
  uint32_t x_abs = xbits.uintval() & FPBits::FloatProp::EXP_MANT_MASK;

  // |x| >= 1.0
  if (LIBC_UNLIKELY(x_abs >= 0x3F80'0000U)) {
    if (xbits.is_nan()) {
      return x;
    }
    // |x| == 0
    if (x_abs == 0x3F80'0000U) {
      fputil::set_except(FE_DIVBYZERO);
      return with_errno(FPBits::inf(sign).get_val(), ERANGE);
    } else {
      fputil::set_except(FE_INVALID);
      return with_errno(
          FPBits::build_nan(1 << (fputil::MantissaWidth<float>::VALUE - 1)),
          EDOM);
    }
  }

  // |x| < ~0.10
  if (LIBC_UNLIKELY(x_abs <= 0x3dcc'0000U)) {
    // |x| <= 2^-26
    if (LIBC_UNLIKELY(x_abs <= 0x3280'0000U)) {
      return LIBC_UNLIKELY(x_abs == 0) ? x
                                       : (x + 0x1.5555555555555p-2 * x * x * x);
    }

    double xdbl = x;
    double x2 = xdbl * xdbl;
    // Pure Taylor series.
    double pe = fputil::polyeval(x2, 0.0, 0x1.5555555555555p-2,
                                 0x1.999999999999ap-3, 0x1.2492492492492p-3,
                                 0x1.c71c71c71c71cp-4, 0x1.745d1745d1746p-4);
    return fputil::multiply_add(xdbl, pe, xdbl);
  }
  double xdbl = x;
  return 0.5 * log_eval((xdbl + 1.0) / (xdbl - 1.0));
}

} // namespace __llvm_libc

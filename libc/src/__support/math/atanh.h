//===-- Implementation header for atanh -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ATANH_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ATANH_H

#include "log1p.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE double atanh(double x) {
  using FPBits = fputil::FPBits<double>;
  FPBits xbits(x);
  Sign sign = xbits.sign();
  uint64_t x_abs_u = xbits.abs().uintval();

  // Handle NaN.
  if (LIBC_UNLIKELY(xbits.is_nan())) {
    if (xbits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // |x| >= 1.
  if (LIBC_UNLIKELY(x_abs_u >= 0x3ff0000000000000ULL)) {
    if (x_abs_u == 0x3ff0000000000000ULL) {
      // |x| == 1: return +/-inf with ERANGE.
      fputil::set_errno_if_required(ERANGE);
      fputil::raise_except_if_required(FE_DIVBYZERO);
      return FPBits::inf(sign).get_val();
    }
    // |x| > 1: domain error.
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  // For very small |x| (|x| <= 2^-27), atanh(x) ~ x.
  // Use FP arithmetic to ensure FTZ/DAZ mode behavior.
  if (LIBC_UNLIKELY(x_abs_u <= 0x3e40000000000000ULL)) {
    // atanh(+/-0) = +/-0, preserve sign of zero exactly.
    if (LIBC_UNLIKELY(x_abs_u == 0))
      return x;
    double x_abs = xbits.abs().get_val();
    double x2 = x_abs * x_abs;
    double result = x_abs + x2 * x_abs * (1.0 / 3.0);
    return sign == Sign::NEG ? -result : result;
  }

  // General case: atanh(x) = 0.5 * log((1+x)/(1-x))
  //                        = 0.5 * log1p(2x / (1-x))
  double x_abs = xbits.abs().get_val();
  double result = 0.5 * math::log1p(2.0 * x_abs / (1.0 - x_abs));
  return sign == Sign::NEG ? -result : result;
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ATANH_H

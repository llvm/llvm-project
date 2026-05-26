//===-- Implementation header for sinh --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_SINH_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_SINH_H

#include "exp.h"
#include "expm1.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE double sinh(double x) {
  using FPBits = fputil::FPBits<double>;
  FPBits xbits(x);
  bool is_neg = xbits.is_neg();
  // Work with |x|.
  xbits.set_sign(Sign::POS);
  double x_abs = xbits.get_val();
  uint64_t x_abs_u = xbits.uintval();

  // Handle NaN: sinh(NaN) = NaN.
  if (LIBC_UNLIKELY(xbits.is_nan())) {
    if (xbits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // sinh(+/-inf) = +/-inf.
  if (LIBC_UNLIKELY(xbits.is_inf()))
    return x;

  // For very small |x| (|x| <= 2^-26), sinh(x) ~ x.
  // Use FP arithmetic to ensure FTZ/DAZ mode behavior.
  if (LIBC_UNLIKELY(x_abs_u <= 0x3e50000000000000ULL)) {
    // sinh(+/-0) = +/-0, preserve sign of zero exactly.
    if (LIBC_UNLIKELY(x_abs_u == 0))
      return x;
    double x2 = x_abs * x_abs;
    double result = x_abs + x2 * x_abs * (1.0 / 6.0);
    return is_neg ? -result : result;
  }

  // For |x| >= 710, overflow.
  if (LIBC_UNLIKELY(x_abs_u >= 0x4086340000000000ULL)) {
    int rounding = fputil::quick_get_round();
    if ((rounding == FE_DOWNWARD && !is_neg) ||
        (rounding == FE_UPWARD && is_neg) || rounding == FE_TOWARDZERO)
      return is_neg ? -FPBits::max_normal().get_val()
                    : FPBits::max_normal().get_val();
    fputil::set_errno_if_required(ERANGE);
    fputil::raise_except_if_required(FE_OVERFLOW);
    return x + (is_neg ? -FPBits::inf().get_val() : FPBits::inf().get_val());
  }

  double result;
  if (x_abs_u < 0x3ff0000000000000ULL) {
    // |x| < 1: use expm1 to avoid catastrophic cancellation.
    // sinh(x) = expm1(x)/2 + expm1(x)/(2*(1+expm1(x)))
    double em1 = math::expm1(x_abs);
    result = em1 / 2.0 + em1 / (2.0 * (1.0 + em1));
  } else {
    // |x| >= 1: sinh(x) = (exp(x) - exp(-x)) / 2.
    double ex = math::exp(x_abs);
    result = (ex - 1.0 / ex) * 0.5;
  }

  return is_neg ? -result : result;
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_SINH_H

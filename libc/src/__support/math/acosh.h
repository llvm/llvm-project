//===-- Implementation header for acosh -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ACOSH_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ACOSH_H

#include "log1p.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE double acosh(double x) {
  using FPBits = fputil::FPBits<double>;
  using DoubleDouble = fputil::DoubleDouble;
  using namespace log1p_internal;

  FPBits xbits(x);

  // x <= 1.0 is false for NaN (NaN comparisons always return false), so NaN
  // falls through to the large-bits branch below where it is handled.
  if (LIBC_UNLIKELY(x <= 1.0)) {
    if (x == 1.0)
      return 0.0;
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  // For x >= 2^52, the argument to log simplifies to 2x:
  //   acosh(x) = log(x + sqrt(x^2 - 1))
  //   sqrt(x^2 - 1) = x * sqrt(1 - 1/x^2)
  //                 = x * (1 - 1/(2x^2) - 1/(8x^4) - ...)
  //   x + sqrt(x^2 - 1) = 2x * (1 - 1/(4x^2) - 1/(16x^4) - ...)
  //   acosh(x) = log(2x) + log(1 - 1/(4x^2) - ...) = log(2x) - 1/(4x^2) - ...
  // At x = 2^52 the dropped term 1/(4x^2) < 2^-106, far below
  // 0.5 * ULP(log(2^52)) ~ 2^-47, so log(2x) is correctly rounded for acosh.
  // For x^2 > ~2^1022 exact_mult(x,x) would overflow, making this redirect
  // necessary for all x >= 2^52. NaN and +inf also have uintval >= 0x4330...
  if (LIBC_UNLIKELY(xbits.uintval() >= 0x4330'0000'0000'0000ULL)) {
    if (LIBC_UNLIKELY(xbits.is_inf_or_nan())) {
      if (xbits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return x;
    }
    // Compute log(2x) = log(x) + log(2) via log_dd_core with e_adj = 1.
    // Passing x rather than 2x avoids overflow for x >= 2^1023 and ensures
    // correct rounding: the Ziv test inside log_dd_core runs on the full
    // double-double sum (including the e_adj * log(2) term) before rounding.
    return log_dd_core({0.0, x}, 1);
  }

  // acosh(x) = log(x + sqrt(x^2 - 1)) for x in (1, 2^52).
  // Compute y = x + sqrt(x^2 - 1) as a double-double, then log(y).

  // x^2 - 1 as a double-double. x > 1 implies x^2 > 1 = |(-1)|, so
  // Fast2Sum (exact_add) is valid.
  DoubleDouble x_sq = fputil::exact_mult(x, x);
  DoubleDouble v_dd = fputil::exact_add(x_sq.hi, -1.0);
  v_dd.lo += x_sq.lo;

  // sqrt(x^2 - 1) as a double-double via one Newton step.
  double s_hi = fputil::sqrt<double>(v_dd.hi);
  double s_inv = 0.5 / s_hi;
  DoubleDouble s_sq = fputil::exact_mult(s_hi, s_hi);
  double s_lo = ((v_dd.hi - s_sq.hi) - s_sq.lo + v_dd.lo) * s_inv;

  // y = x + sqrt(x^2 - 1) as a double-double. Fast2Sum applies since
  // sqrt(x^2 - 1) < x for all x > 1.
  DoubleDouble y_dd = fputil::exact_add(x, s_hi);
  double y_lo = y_dd.lo + s_lo;

  return log_dd_core({y_lo, y_dd.hi}, 0);
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ACOSH_H

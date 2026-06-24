//===-- Implementation header for asinh -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ASINH_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ASINH_H

#include "log.h"
#include "log1p.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE double asinh(double x) {
  using FPBits = fputil::FPBits<double>;
  FPBits xbits(x);

  // Handle NaN.
  if (LIBC_UNLIKELY(xbits.is_nan())) {
    if (xbits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // Handle +/-inf: asinh(+/-inf) = +/-inf.
  if (LIBC_UNLIKELY(xbits.is_inf()))
    return x;

  uint64_t x_abs_u = xbits.abs().uintval();
  double x_abs = xbits.abs().get_val();
  bool is_neg = xbits.is_neg();

  // For very small |x| (|x| <= 2^-26), asinh(x) ~ x.
  // Use FP arithmetic to ensure FTZ/DAZ mode behavior.
  if (LIBC_UNLIKELY(x_abs_u <= 0x3e50000000000000ULL)) {
    // asinh(+/-0) = +/-0, preserve sign of zero exactly.
    if (LIBC_UNLIKELY(x_abs_u == 0))
      return x;
    double x2 = x_abs * x_abs;
    double result = x_abs - x2 * x_abs * (1.0 / 6.0);
    return is_neg ? -result : result;
  }

  double result;
  // For large |x| (|x| >= 2^28): asinh(x) ~ log(2|x|) = log(|x|) + log(2).
  if (LIBC_UNLIKELY(x_abs_u >= 0x41b0000000000000ULL)) {
    constexpr double LOG_2 = 0x1.62e42fefa39efp-1;
    result = math::log(x_abs) + LOG_2;
  } else if (x_abs_u >= 0x3fe0000000000000ULL) {
    // |x| >= 0.5: asinh(x) = log(x + sqrt(x^2 + 1)).
    result = math::log(x_abs + fputil::sqrt<double>(x_abs * x_abs + 1.0));
  } else {
    // |x| < 0.5: use log1p for better accuracy near 0.
    // asinh(x) = log1p(x + x^2 / (1 + sqrt(1 + x^2)))
    double x2 = x_abs * x_abs;
    double sqrt1px2 = fputil::sqrt<double>(1.0 + x2);
    result = math::log1p(x_abs + x2 / (1.0 + sqrt1px2));
  }

  return is_neg ? -result : result;
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ASINH_H

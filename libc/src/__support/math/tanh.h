//===-- Implementation header for tanh --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_TANH_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_TANH_H

#include "expm1.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE double tanh(double x) {
  using FPBits = fputil::FPBits<double>;
  FPBits xbits(x);
  bool is_neg = xbits.is_neg();
  // Work with |x|.
  xbits.set_sign(Sign::POS);
  double x_abs = xbits.get_val();
  uint64_t x_abs_u = xbits.uintval();

  // Handle NaN: tanh(NaN) = NaN.
  if (LIBC_UNLIKELY(xbits.is_nan())) {
    if (xbits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // For very small |x| (|x| <= 2^-27), tanh(x) ~ x.
  // Use x - x^3/3 to ensure FP operations are performed (for FTZ/DAZ modes).
  if (LIBC_UNLIKELY(x_abs_u <= 0x3e40000000000000ULL)) {
    // tanh(+/-0) = +/-0, preserve sign of zero exactly.
    if (LIBC_UNLIKELY(x_abs_u == 0))
      return x;
    double x2 = x_abs * x_abs;
    double result = x_abs - x2 * x_abs * (1.0 / 3.0);
    return is_neg ? -result : result;
  }

  // For |x| >= 20, tanh(x) ~ sign(x) * 1.
  // (e^40 - 1)/(e^40 + 1) rounds to 1 in double precision.
  if (LIBC_UNLIKELY(x_abs_u >= 0x4034000000000000ULL)) {
    // tanh(+/-inf) = +/-1.
    return is_neg ? -1.0 : 1.0;
  }

  // General case: tanh(x) = expm1(2x) / (expm1(2x) + 2).
  double e2xm1 = math::expm1(2.0 * x_abs);
  double result = e2xm1 / (e2xm1 + 2.0);
  return is_neg ? -result : result;
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_TANH_H

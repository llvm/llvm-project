//===-- Implementation header for acosh -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ACOSH_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ACOSH_H

#include "log.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE double acosh(double x) {
  using FPBits = fputil::FPBits<double>;
  FPBits xbits(x);
  uint64_t x_u = xbits.uintval();

  // Handle NaN.
  if (LIBC_UNLIKELY(xbits.is_nan())) {
    if (xbits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // acosh(+inf) = +inf.
  if (LIBC_UNLIKELY(xbits.is_inf()))
    return x;

  // Domain error: acosh(x) is undefined for x < 1.
  if (LIBC_UNLIKELY(x_u < 0x3ff0000000000000ULL)) {
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  // acosh(1) = 0.
  if (LIBC_UNLIKELY(x_u == 0x3ff0000000000000ULL))
    return 0.0;

  // For large x (x >= 2^28): acosh(x) ~ log(2x) = log(x) + log(2).
  if (LIBC_UNLIKELY(x_u >= 0x41b0000000000000ULL)) {
    constexpr double LOG_2 = 0x1.62e42fefa39efp-1;
    return math::log(x) + LOG_2;
  }

  // General case: acosh(x) = log(x + sqrt(x^2 - 1)).
  double x2m1 = x * x - 1.0;
  return math::log(x + fputil::sqrt<double>(x2m1));
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ACOSH_H

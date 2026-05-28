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
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE double acosh(double x) {
  using FPBits = fputil::FPBits<double>;
  FPBits xbits(x);
  uint64_t x_u = xbits.uintval();

  // acosh is defined only for x >= 1.  The comparison x <= 1.0 is false for
  // NaN, so NaN falls through to the large-x / NaN path below.
  if (LIBC_UNLIKELY(x <= 1.0)) {
    if (x == 1.0)
      return 0.0;
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  // For large x (x >= 2^28) and for NaN / +inf.
  if (LIBC_UNLIKELY(x_u >= 0x41b0000000000000ULL)) {
    if (LIBC_UNLIKELY(xbits.is_inf_or_nan())) {
      if (xbits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return x; // +inf (negative inf is excluded by x <= 1.0 above)
    }
    // acosh(x) = log(2x) + O(1/x^2); for x >= 2^28 the correction is < 0.5 ULP.
    constexpr double LOG_2 = 0x1.62e42fefa39efp-1;
    return math::log(x) + LOG_2;
  }

  // General case: acosh(x) = log(x + sqrt(x^2 - 1)).
  double x2m1 = fputil::multiply_add(x, x, -1.0);
  return math::log(x + fputil::sqrt<double>(x2m1));
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ACOSH_H

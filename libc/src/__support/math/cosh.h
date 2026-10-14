//===-- Implementation header for cosh --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_COSH_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_COSH_H

#include "exp.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE double cosh(double x) {
  using FPBits = fputil::FPBits<double>;
  FPBits xbits(x);
  // cosh is even, work with |x|.
  xbits.set_sign(Sign::POS);
  double x_abs = xbits.get_val();
  uint64_t x_abs_u = xbits.uintval();

  // Handle NaN: cosh(NaN) = NaN.
  if (LIBC_UNLIKELY(xbits.is_nan())) {
    if (xbits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // cosh(+/-inf) = +inf.
  if (LIBC_UNLIKELY(xbits.is_inf()))
    return FPBits::inf(Sign::POS).get_val();

  // For very small |x| (|x| <= 2^-27), cosh(x) ~ 1.
  if (LIBC_UNLIKELY(x_abs_u <= 0x3e40000000000000ULL))
    return 1.0;

  // For |x| >= 710, overflow.
  if (LIBC_UNLIKELY(x_abs_u >= 0x4086340000000000ULL)) {
    int rounding = fputil::quick_get_round();
    if (rounding == FE_DOWNWARD || rounding == FE_TOWARDZERO)
      return FPBits::max_normal().get_val();
    fputil::set_errno_if_required(ERANGE);
    fputil::raise_except_if_required(FE_OVERFLOW);
    return FPBits::inf(Sign::POS).get_val();
  }

  // General case: cosh(x) = (exp(x) + exp(-x)) / 2.
  double ex = math::exp(x_abs);
  return (ex + 1.0 / ex) * 0.5;
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_COSH_H

//===-- Double-precision tgamma function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_TGAMMA_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_TGAMMA_H

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE double tgamma(double x) {
  using FPBits = fputil::FPBits<double>;
  FPBits xbits(x);

  if (LIBC_UNLIKELY(xbits.is_inf_or_nan())) {
    if (xbits.is_nan()) {
      if (xbits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return x;
    }
    // tgamma(-inf) = NaN with domain error
    if (xbits.is_neg()) {
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    // tgamma(+inf) = +inf.
    return x;
  }

  // Gamma has a pole at 0, with sign following the input
  if (LIBC_UNLIKELY(x == 0.0)) {
    fputil::set_errno_if_required(ERANGE);
    fputil::raise_except_if_required(FE_DIVBYZERO);
    return FPBits::inf(xbits.sign()).get_val();
  }

  // tgamma(x) > DBL_MAX for x >= 0x1.573fae561f648p+7
  // Source: https://members.loria.fr/PZimmermann/papers/gamma.pdf
  if (LIBC_UNLIKELY(x >= 0x1.573fae561f648p+7)) {
    fputil::set_errno_if_required(ERANGE);
    fputil::raise_except_if_required(FE_OVERFLOW);
    return FPBits::inf().get_val();
  }

  // Gamma has poles at all negative integers
  if (LIBC_UNLIKELY(x < 0.0 && fputil::nearest_integer(x) == x)) {
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  // TODO: Implement tgamma for the remaining input ranges
  return 0.0;
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_TGAMMA_H

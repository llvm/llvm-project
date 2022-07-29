//===-- Single-precision cosh function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/coshf.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/math/generic/expxf.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(float, coshf, (float x)) {
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);
  xbits.set_sign(false);
  x = xbits.get_val();

  uint32_t x_u = xbits.uintval();

  // |x| <= 2^-26
  if (unlikely(x_u <= 0x3280'0000U)) {
    return 1.0f + x;
  }

  // When |x| >= 90, or x is inf or nan
  if (unlikely(x_u >= 0x42b4'0000U)) {
    if (xbits.is_inf_or_nan())
      return x + FPBits::inf().get_val();

    int rounding = fputil::get_round();
    if (unlikely(rounding == FE_DOWNWARD || rounding == FE_TOWARDZERO))
      return FPBits(FPBits::MAX_NORMAL).get_val();

    errno = ERANGE;

    return x + FPBits::inf().get_val();
  }
  auto ep_p = exp_eval<-1>(x);
  auto ep_m = exp_eval<-1>(-x);
  // 0.5 * exp(x)  = ep_p.mult_exp * (ep_p.r + 1)
  //               = ep_p.mult_exp * ep_p.r + ep_p.mult_exp
  // 0.5 * exp(-x) = ep_m.mult_exp * (ep_m.r + 1)
  //               = ep_m.mult_exp * ep_m.r + ep_m.mult_exp
  // cos(x) = 0.5 * exp(x) + 0.5 * expm1(-x)
  double ep = fputil::multiply_add(ep_p.mult_exp, ep_p.r, ep_p.mult_exp) +
              fputil::multiply_add(ep_m.mult_exp, ep_m.r, ep_m.mult_exp);
  return ep;
}

} // namespace __llvm_libc

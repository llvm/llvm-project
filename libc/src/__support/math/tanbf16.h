//===-- Implementation of tanbf16 function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_TANBF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_TANBF16_H

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "sincosf16_utils.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE bfloat16 tanbf16(bfloat16 x) {
  using namespace sincosf16_internal;
  using FPBits = fputil::FPBits<bfloat16>;
  FPBits xbits(x);

  uint16_t x_u = xbits.uintval();
  uint16_t x_abs = x_u & 0x7fff;
  float xf = x;

  // NaN or -/+ INF
  if (x_abs >= 0x7F80) {
    // NaN
    if (xbits.is_nan()) {
      if (xbits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return x;
    }
    //|x| is +/- INF
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return x + FPBits::quiet_nan().get_val();
  }

  // |x| = {0}
  if (LIBC_UNLIKELY(x_abs == 0)) {
    return x;
  }

  // Through Exhaustive testing
  // The last value where tan(x) ~ x is 0x3db8
  if (LIBC_UNLIKELY(x_abs <= 0x3db8)) {
    int rounding = fputil::quick_get_round();
    // separate case handles it with magnitude of 2^-13
    if ((xbits.is_pos() && rounding == FE_UPWARD) ||
        (xbits.is_neg() && rounding == FE_DOWNWARD))
      return fputil::cast<bfloat16>(fputil::multiply_add(xf, 0x1.0p-13f, xf));
    return x;
  }

  float sin_k, cos_k, sin_y, cosm1_y;
  sincosf16_eval(xf, sin_k, cos_k, sin_y, cosm1_y);

  using fputil::multiply_add;
  return fputil::cast<bfloat16>(
      multiply_add(sin_y, cos_k, multiply_add(cosm1_y, sin_k, sin_k)) /
      multiply_add(sin_y, -sin_k, multiply_add(cosm1_y, cos_k, cos_k)));
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_TANBF16_H

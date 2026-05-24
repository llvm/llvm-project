//===-- Implementation header for acospibf16 --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ACOSPIBF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ACOSPIBF16_H

#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/inv_trigf_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

LIBC_INLINE bfloat16 acospibf16(bfloat16 x) {

  using FPBits = fputil::FPBits<bfloat16>;
  FPBits xbits(x);

  uint16_t x_u = xbits.uintval();
  uint16_t x_abs = x_u & 0x7fff;
  bool sign = (x_u >> 15);
  float xf = x;

  float xf_abs = (xf < 0 ? -xf : xf);
  float x_sq = xf_abs * xf_abs;

  // case 1: |x| <= 0.5
  if (x_abs <= 0x3F00) {
    // |x| = {0}
    if (LIBC_UNLIKELY(x_abs == 0))
      return fputil::cast<bfloat16>(0.5f);

    float xp = fputil::cast<float>(inv_trigf_utils_internal::asinpi_eval(x_sq));
    float result =
        xf *
        fputil::multiply_add(
            x_sq, xp,
            fputil::cast<float>(inv_trigf_utils_internal::ASINPI_COEFFS[0]));
    return fputil::cast<bfloat16>(0.5 - result);
  }

  // case 2: 0.5< |x|<= 1.0
  if (x_abs <= 0x3F80) {
    // |x| = {1}
    if (x_abs == 0x3F80) {
      if (sign)
        return fputil::cast<bfloat16>(1.0f);
      else
        return fputil::cast<bfloat16>(0.0f);
    }

    // using reduction for acos:
    // f(x) = acos(|x|) = 2*asin(sqrt((1 - |x|)/2)),
    // then for
    // using same for acospi
    // x>=0: f(x) = acos(x)/pi  = acospi(x)
    // x<0 : f(x) = (pi - acos(x))/pi = 1 - acospi(x)

    float t = fputil::multiply_add<float>(xf_abs, -0.5f, 0.5f);
    float t_sqrt = fputil::sqrt<float>(t);
    float tp = fputil::cast<float>(inv_trigf_utils_internal::asinpi_eval(t));
    float asin_sqrt_t =
        t_sqrt *
        (fputil::multiply_add(
            t, tp,
            fputil::cast<float>(inv_trigf_utils_internal::ASINPI_COEFFS[0])));

    return fputil::cast<bfloat16>(
        (sign) ? fputil::multiply_add(asin_sqrt_t, -2.0f, 1.0f)
               : 2 * asin_sqrt_t);
  }
  // case 3: NaN or Inf
  // NaN
  if (xbits.is_nan()) {
    if (xbits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x; // quiet NaN
  }
  // inf
  fputil::raise_except_if_required(FE_INVALID);
  fputil::set_errno_if_required(EDOM); // Domain is bounded
  return FPBits::quiet_nan().get_val();
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ACOSPIBF16_H

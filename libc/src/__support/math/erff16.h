//===-- Implementation header for erff16 ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ERFF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ERFF16_H

#include "include/llvm-libc-macros/float16-macros.h"

#ifdef LIBC_TYPES_HAS_FLOAT16

#include "common_constants.h" // ERFF_COEFFS
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE float16 erff16(float16 x) {
  using namespace common_constants_internal;

  using FPBits = typename fputil::FPBits<float16>;
  FPBits xbits(x);
  uint16_t x_abs = xbits.abs().uintval();

  // |x| >= 4.0
  if (LIBC_UNLIKELY(x_abs >= 0x4400U)) {
    // Check for NaN or Inf
    if (LIBC_UNLIKELY(x_abs >= 0x7c00U)) {
      if (x_abs > 0x7c00U) {
        if (xbits.is_signaling_nan()) {
          fputil::raise_except_if_required(FE_INVALID);
          return FPBits::quiet_nan().get_val();
        }
        return x;
      }
      // Inf -> returns 1.0 or -1.0
      return xbits.is_neg() ? -1.0f16 : 1.0f16;
    }

    return static_cast<float16>(xbits.is_neg() ? -1.0 - x * 0x1.0p-50
                                               : 1.0 - x * 0x1.0p-50);
  }

  // Polynomial approximation:
  //   erf(x) ~ x * (c0 + c1 * x^2 + c2 * x^4 + ... + c7 * x^14)

  using FPBits32 = typename fputil::FPBits<float>;
  float xf = x;
  FPBits32 xbits32(xf);
  uint32_t x_abs32 = xbits32.abs().uintval();

  constexpr uint32_t EIGHT = 3 << FPBits32::FRACTION_LEN;
  int idx = static_cast<int>(FPBits32(x_abs32 + EIGHT).get_val());

  double xd = static_cast<double>(x);
  double xsq = xd * xd;

  double x4 = xsq * xsq;
  double c0 =
      fputil::multiply_add(xsq, ERFF_COEFFS[idx][1], ERFF_COEFFS[idx][0]);
  double c1 =
      fputil::multiply_add(xsq, ERFF_COEFFS[idx][3], ERFF_COEFFS[idx][2]);
  double c2 =
      fputil::multiply_add(xsq, ERFF_COEFFS[idx][5], ERFF_COEFFS[idx][4]);
  double c3 =
      fputil::multiply_add(xsq, ERFF_COEFFS[idx][7], ERFF_COEFFS[idx][6]);

  double x8 = x4 * x4;
  double p0 = fputil::multiply_add(x4, c1, c0);
  double p1 = fputil::multiply_add(x4, c3, c2);

  return static_cast<float16>(xd * fputil::multiply_add(x8, p1, p0));
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TYPES_HAS_FLOAT16

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ERFF16_H

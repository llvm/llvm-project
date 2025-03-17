//===-- Implementation of atanh(x) function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atanhf16.h"
#include "explogxf.h"
#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, atanhf16, (float16 x)) {
  using FPBits = typename fputil::FPBits<float16>;

  FPBits xbits(x);
  Sign sign = xbits.sign();
  uint16_t x_abs = xbits.abs().uintval();

  if (LIBC_UNLIKELY(x_abs >= 0x3c00U)) {
    if (xbits.is_nan()) {
      return x;
    }
    // |x| == 1.0
    if (x_abs == 0x3c00U) {
      fputil::set_errno_if_required(ERANGE);
      fputil::raise_except_if_required(FE_DIVBYZERO);
      return FPBits::inf(sign).get_val();
    } else {
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
  }

  // For |x| less than approximately 0.10
  if (LIBC_UNLIKELY(x_abs <= 0x2e66U)) {
    // The Taylor expansion of atanh(x) is:
    //    atanh(x) = x + x^3/3 + x^5/5 + x^7/7 + x^9/9 + x^11/11
    //             = x * [1 + x^2/3 + x^4/5 + x^6/7 + x^8/9 + x^10/11]
    // When |x| < 0x0100U, this can be approximated by:
    //    atanh(x) ≈ x + (1/3)*x^3
    if (LIBC_UNLIKELY(x_abs < 0x0100U)) {
      return static_cast<float16>(
          LIBC_UNLIKELY(x_abs == 0) ? x : (x + 0x1.555556p-2 * x * x * x));
    }

    // For 0x0100U <= |x| <= 0x2e66U:
    //   Let t = x^2.
    //   Define P(t) ≈ (1/3)*t + (1/5)*t^2 + (1/7)*t^3 + (1/9)*t^4 + (1/11)*t^5.
    // The coefficients below were derived using Sollya:
    //   > display = hexadecimal;
    //   > round(1/3, SG, RN);
    //   > round(1/5, SG, RN);
    //   > round(1/7, SG, RN);
    //   > round(1/9, SG, RN);
    //   > round(1/11, SG, RN);
    // This yields:
    //   0x1.555556p-2
    //   0x1.99999ap-3
    //   0x1.24924ap-3
    //   0x1.c71c72p-4
    //   0x1.745d18p-4f
    // Thus, atanh(x) ≈ x * (1 + P(x^2)).
    float xf = x;
    float x2 = xf * xf;
    float pe = fputil::polyeval(x2, 0.0f, 0x1.555556p-2f, 0x1.99999ap-3f,
                                0x1.24924ap-3f, 0x1.c71c72p-4f, 0x1.745d18p-4f);
    return static_cast<float16>(fputil::multiply_add(xf, pe, xf));
  }

  float xf = x;
  return static_cast<float16>(0.5 * log_eval((xf + 1.0) / (xf - 1.0)));
}

} // namespace LIBC_NAMESPACE_DECL

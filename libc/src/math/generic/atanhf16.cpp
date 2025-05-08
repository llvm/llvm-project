//===-- Half-precision atanh(x) function ----------------------------------===//
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
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/except_value_utils.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

static constexpr size_t N_EXCEPTS = 1;
static constexpr fputil::ExceptValues<float16, N_EXCEPTS> ATANHF16_EXCEPTS{{
    // (input, RZ output, RU offset, RD offset, RN offset)
    // x = 0x1.a5cp-4, atanhf16(x) = 0x1.a74p-4 (RZ)
    {0x2E97, 0x2E9D, 1, 0, 0},
}};

LLVM_LIBC_FUNCTION(float16, atanhf16, (float16 x)) {
  using FPBits = fputil::FPBits<float16>;

  FPBits xbits(x);
  Sign sign = xbits.sign();
  uint16_t x_abs = xbits.abs().uintval();

  // |x| >= 1
  if (LIBC_UNLIKELY(x_abs >= 0x3c00U)) {
    if (xbits.is_nan()) {
      if (xbits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return x;
    }

    // |x| == 1.0
    if (x_abs == 0x3c00U) {
      fputil::set_errno_if_required(ERANGE);
      fputil::raise_except_if_required(FE_DIVBYZERO);
      return FPBits::inf(sign).get_val();
    }
    // |x| > 1.0
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  if (auto r = ATANHF16_EXCEPTS.lookup(xbits.uintval());
      LIBC_UNLIKELY(r.has_value()))
    return r.value();

  // For |x| less than approximately 0.24
  if (LIBC_UNLIKELY(x_abs <= 0x33f3U)) {
    // atanh(+/-0) = +/-0
    if (LIBC_UNLIKELY(x_abs == 0U))
      return x;
    // The Taylor expansion of atanh(x) is:
    //    atanh(x) = x + x^3/3 + x^5/5 + x^7/7 + x^9/9 + x^11/11
    //             = x * [1 + x^2/3 + x^4/5 + x^6/7 + x^8/9 + x^10/11]
    // When |x| < 2^-5 (0x0800U), this can be approximated by:
    //    atanh(x) ≈ x + (1/3)*x^3
    if (LIBC_UNLIKELY(x_abs < 0x0800U)) {
      float xf = x;
      return fputil::cast<float16>(xf + 0x1.555556p-2f * xf * xf * xf);
    }

    // For 2^-5 <= |x| <= 0x1.fccp-3 (~0.24):
    //   Let t = x^2.
    //   Define P(t) ≈ (1/3)*t + (1/5)*t^2 + (1/7)*t^3 + (1/9)*t^4 + (1/11)*t^5.
    // Coefficients (from Sollya, RN, hexadecimal):
    //  1/3 = 0x1.555556p-2, 1/5 = 0x1.99999ap-3, 1/7 = 0x1.24924ap-3,
    //  1/9 = 0x1.c71c72p-4, 1/11 = 0x1.745d18p-4
    // Thus, atanh(x) ≈ x * (1 + P(x^2)).
    float xf = x;
    float x2 = xf * xf;
    float pe = fputil::polyeval(x2, 0.0f, 0x1.555556p-2f, 0x1.99999ap-3f,
                                0x1.24924ap-3f, 0x1.c71c72p-4f, 0x1.745d18p-4f);
    return fputil::cast<float16>(fputil::multiply_add(xf, pe, xf));
  }

  float xf = x;
  return fputil::cast<float16>(0.5 * log_eval_f((xf + 1.0f) / (xf - 1.0f)));
}

} // namespace LIBC_NAMESPACE_DECL

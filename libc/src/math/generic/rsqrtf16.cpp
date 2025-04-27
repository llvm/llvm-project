//===-- Half-precision rsqrt function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "src/math/rsqrtf16.h"
#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, rsqrtf16, (float16 x)) {
  using FPBits = fputil::FPBits<float16>;
  FPBits xbits(x);

  uint16_t x_u = xbits.uintval();
  uint16_t x_abs = x_u & 0x7fff;
  uint16_t x_sign = x_u >> 15;

  // x is NaN
  if (LIBC_UNLIKELY(xbits.is_nan())) {
    if (xbits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // |x| = 0
  if (LIBC_UNLIKELY(x_abs == 0x0)) {
    fputil::raise_except_if_required(FE_DIVBYZERO);
    fputil::set_errno_if_required(ERANGE);
    return FPBits::quiet_nan().get_val();
  }

  // -inf <= x < 0
  if (LIBC_UNLIKELY(x_sign == 1)) {
    fputil::raise_except_if_required(FE_INVALID);
    fputil::set_errno_if_required(EDOM);
    return FPBits::quiet_nan().get_val();
  }

  // x = +inf => rsqrt(x) = 0
  if (LIBC_UNLIKELY(xbits.is_inf())) {
    return fputil::cast<float16>(0.0f);
  }

  // x = 1 => rsqrt(x) = 1
  if (LIBC_UNLIKELY(x_u == 0x1)) {
    return fputil::cast<float16>(1.0f);
  }

  // x is valid, estimate the result
  // 3-degree polynomial generated using Sollya
  // P = fpminimax(1/sqrt(x), [|1, 2, 3|], [|SG...|], [0.5, 1]);
  float xf = x;
  float result =
      fputil::polyeval(xf, 0x1.d42408p2f, -0x1.7cc4fep3f, 0x1.66cb6ap2f);
  return fputil::cast<float16>(result);
}
} // namespace LIBC_NAMESPACE_DECL

//===-- Single-precision 2^x function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/exp2f.h"
#include "common_constants.h"
#include "explogxf.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/__support/common.h"

#include <errno.h>

namespace __llvm_libc {

constexpr uint32_t exval1 = 0x3b42'9d37U;
constexpr uint32_t exval2 = 0xbcf3'a937U;
constexpr uint32_t exval_mask = exval1 & exval2;

LLVM_LIBC_FUNCTION(float, exp2f, (float x)) {
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);

  uint32_t x_u = xbits.uintval();
  uint32_t x_abs = x_u & 0x7fff'ffffU;

  // // When |x| >= 128, |x| < 2^-25, or x is nan
  if (unlikely(x_abs >= 0x4300'0000U || x_abs <= 0x3280'0000U)) {
    // |x| < 2^-25
    if (x_abs <= 0x3280'0000U) {
      return 1.0f + x;
    }
    // x >= 128
    if (!xbits.get_sign()) {
      // x is finite
      if (x_u < 0x7f80'0000U) {
        int rounding = fputil::get_round();
        if (rounding == FE_DOWNWARD || rounding == FE_TOWARDZERO)
          return static_cast<float>(FPBits(FPBits::MAX_NORMAL));

        errno = ERANGE;
      }
      // x is +inf or nan
      return x + FPBits::inf().get_val();
    }
    // x < -150
    if (x_u >= 0xc316'0000U) {
      // exp(-Inf) = 0
      if (xbits.is_inf())
        return 0.0f;
      // exp(nan) = nan
      if (xbits.is_nan())
        return x;
      if (fputil::get_round() == FE_UPWARD)
        return FPBits(FPBits::MIN_SUBNORMAL).get_val();
      if (x != 0.0f)
        errno = ERANGE;
      return 0.0f;
    }
  }

  if (unlikely(x_u & exval_mask) == exval_mask) {
    if (unlikely(x_u == exval1)) { // x = 0x1.853a6ep-9f
      if (fputil::get_round() == FE_TONEAREST)
        return 0x1.00870ap+0f;
    } else if (unlikely(x_u == exval2)) { // x = -0x1.e7526ep-6f
      if (fputil::get_round() == FE_TONEAREST)
        return 0x1.f58d62p-1f;
    }
  }

  return exp2_eval(x);
}

} // namespace __llvm_libc

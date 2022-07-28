//===-- Single-precision 2^x function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/exp2f.h"
#include "common_constants.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/nearest_integer.h"
#include "src/__support/common.h"

#include <errno.h>

namespace __llvm_libc {

constexpr float mlp = EXP_num_p;
constexpr float mmld = -1.0 / mlp;

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

  float kf = fputil::nearest_integer(x * mlp);
  double dx = fputil::multiply_add(mmld, kf, x);
  double mult_f, ml;
  {
    uint32_t ps = static_cast<int>(kf) + (1 << (EXP_bits_p - 1)) +
                  (fputil::FPBits<double>::EXPONENT_BIAS << EXP_bits_p);
    fputil::FPBits<double> bs;
    bs.set_unbiased_exponent(ps >> EXP_bits_p);
    ml = 1.0 + EXP_2_POW[ps & (EXP_num_p - 1)];
    mult_f = bs.get_val();
  }

  // N[Table[Ln[2]^n/n!,{n,1,6}],30]
  double pe = fputil::polyeval(
      dx, 1.0, 0x1.62e42fefa39efp-1, 0x1.ebfbdff82c58fp-3, 0x1.c6b08d704a0c0p-5,
      0x1.3b2ab6fba4e77p-7, 0x1.5d87fe78a6731p-10, 0x1.430912f86c787p-13);

  return mult_f * ml * pe;
}

} // namespace __llvm_libc

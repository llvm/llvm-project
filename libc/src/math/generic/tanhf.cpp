//===-- Single-precision tanh function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/tanhf.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/properties/cpu_features.h"
#include "src/math/generic/explogxf.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(float, tanhf, (float x)) {
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);
  bool sign = xbits.get_sign();
  uint32_t x_abs = xbits.uintval() & FPBits::FloatProp::EXP_MANT_MASK;

  // |x| <= 2^-26
  if (LIBC_UNLIKELY(x_abs <= 0x3280'0000U)) {
    return LIBC_UNLIKELY(x_abs == 0) ? x
                                     : (x - 0x1.5555555555555p-2 * x * x * x);
  }

  // When |x| >= 15, or x is inf or nan
  if (LIBC_UNLIKELY(x_abs >= 0x4170'0000U)) {
    if (xbits.is_nan())
      return x + 1.0f; // sNaN to qNaN + signal

    if (xbits.is_inf())
      return sign ? -1.0f : 1.0f;

    if (sign) {
      return -1.0f + opt_barrier(FPBits(FPBits::MIN_NORMAL).get_val());
    } else
      return 1.0f - opt_barrier(FPBits(FPBits::MIN_NORMAL).get_val());
  }

  // |x| <= 0.078125
  if (LIBC_UNLIKELY(x_abs <= 0x3da0'0000U)) {
    double xdbl = x;
    double x2 = xdbl * xdbl;
    // Pure Taylor series.
    double pe = fputil::polyeval(x2, 0.0, -0x1.5555555555555p-2,
                                 0x1.1111111111111p-3, -0x1.ba1ba1ba1ba1cp-5,
                                 0x1.664f4882c10fap-6, -0x1.226e355e6c23dp-7);
    return fputil::multiply_add(xdbl, pe, xdbl);
  }

  if (LIBC_UNLIKELY(xbits.bits == 0x4058'e0a3U)) {
    if (fputil::get_round() == FE_DOWNWARD)
      return FPBits(0x3f7f'6ad9U).get_val();
  }

  // Range reduction: e^(2x) = 2^(mid + hi) * e^lo
  auto ep = exp_b_range_reduc<ExpBase>(2.0f * x); // exp(2 * x)
  double r = ExpBase::powb_lo(ep.lo);
  // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
#if defined(LIBC_TARGET_CPU_HAS_FMA)
  return fputil::multiply_add(ep.mh, r, -1.0) /
         fputil::multiply_add(ep.mh, r, 1.0);
#else
  double exp_x = ep.mh * r;
  return (exp_x - 1.0) / (exp_x + 1.0);
#endif // LIBC_TARGET_CPU_HAS_FMA
}

} // namespace __llvm_libc

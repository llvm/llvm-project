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
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/math/generic/explogxf.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float, coshf, (float x)) {
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);
  xbits.set_sign(false);
  x = xbits.get_val();

  uint32_t x_u = xbits.uintval();

  // When |x| >= 90, or x is inf or nan
  if (LIBC_UNLIKELY(x_u >= 0x42b4'0000U || x_u <= 0x3280'0000U)) {
    // |x| <= 2^-26
    if (x_u <= 0x3280'0000U) {
      return 1.0f + x;
    }

    if (xbits.is_inf_or_nan())
      return x + FPBits::inf().get_val();

    int rounding = fputil::quick_get_round();
    if (LIBC_UNLIKELY(rounding == FE_DOWNWARD || rounding == FE_TOWARDZERO))
      return FPBits(FPBits::MAX_NORMAL).get_val();

    fputil::set_errno_if_required(ERANGE);
    fputil::raise_except_if_required(FE_OVERFLOW);

    return x + FPBits::inf().get_val();
  }

  // TODO: We should be able to reduce the latency and reciprocal throughput
  // further by using a low degree (maybe 3-7 ?) minimax polynomial for small
  // but not too small inputs, such as |x| < 2^-2, or |x| < 2^-3.

  // cosh(x) = (e^x + e^(-x)) / 2.
  return static_cast<float>(exp_pm_eval</*is_sinh*/ false>(x));
}

} // namespace LIBC_NAMESPACE

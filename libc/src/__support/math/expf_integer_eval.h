//===-- Implementation header for expf using integer-only --------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H

// TODO: clean up includes
#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/frac128.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/exp_integer_utils.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

namespace integer_only {

LIBC_INLINE float expf(float x) {
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);

  bool is_neg = xbits.is_neg();
  uint16_t x_e = xbits.get_biased_exponent();
  // uint64_t x_u = xbits.get_mantissa();

  // Exceptional values
  // TODO: optimize the branching order
  if (xbits.is_zero()) {
    // e^0 = 1 (exact), for both +/-0
    return 1.0f;
  } else {
    // x is inf or NaN
    if (LIBC_UNLIKELY(x_e > 2 * FPBits::EXP_BIAS)) {
      // e^NaN = NaN
      if (xbits.is_signaling_nan()) {
        // Per conversation with lntue, we don't need to raise exception here,
        // as we're assuming no FPUs/fenv in this kind of environment

        // silencing
        return FPBits::quiet_nan().get_val();
      }

      // e^-inf = 0
      // e^+inf = +inf
      if (xbits.is_inf()) {
        return is_neg ? 0.0f : FPBits::inf().get_val();
      }

      // x is a quiet NaN
      return x;
    } else {
      // TODO: out-of-range (overflow/underflow), exeecute range reduction
    }
  }

  // TODO: execute the normal exp here
  return 0.0f;
}

} // namespace integer_only

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H

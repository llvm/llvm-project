//===-- Implementation header for hypotf ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_HYPOTF_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_HYPOTF_H

#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE float hypotf(float x, float y) {
  using DoubleBits = fputil::FPBits<double>;
  using FPBits = fputil::FPBits<float>;
  using fputil::DoubleDouble;

  uint32_t x_a = FPBits(x).uintval() & 0x7fff'ffff;
  uint32_t y_a = FPBits(y).uintval() & 0x7fff'ffff;

  float x_abs = FPBits(x_a).get_val();
  float y_abs = FPBits(y_a).get_val();

  // Note: replacing `x_a >= FPBits::EXP_MASK` with `x_bits.is_inf_or_nan()`
  // generates extra exponent bit masking instructions on x86-64.
  if (LIBC_UNLIKELY(x_a >= FPBits::EXP_MASK || y_a >= FPBits::EXP_MASK)) {
    // x or y is inf or nan
    FPBits x_bits(x);
    FPBits y_bits(y);
    if (x_bits.is_signaling_nan() || y_bits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    if (x_bits.is_inf() || y_bits.is_inf())
      return FPBits::inf().get_val();
    return x + y;
  }

  bool x_abs_larger = y_abs < x_abs;

  float a = x_abs_larger ? x_abs : y_abs;
  float b = x_abs_larger ? y_abs : x_abs;

  double ad = static_cast<double>(a);
  double bd = static_cast<double>(b);

  // These squares are exact.
  double a_sq = ad * ad;

  DoubleDouble sum_sq;
#ifdef LIBC_TARGET_CPU_HAS_FMA_DOUBLE
  sum_sq.hi = fputil::multiply_add(bd, bd, a_sq);
  sum_sq.lo = fputil::multiply_add(bd, bd, a_sq - sum_sq.hi);
#else
  double b_sq = bd * bd;
  sum_sq = fputil::exact_add(a_sq, b_sq);
#endif

  // Take sqrt in double precision.
  DoubleBits result(fputil::sqrt<double>(sum_sq.hi));

  // We only need to update the result if the sum of squares exceed double
  // precision.
  if (LIBC_UNLIKELY(sum_sq.lo != 0.0)) {
    double r_d = result.get_val();

    // Perform rounding correction.
#ifdef LIBC_TARGET_CPU_HAS_FMA_DOUBLE
    double err = sum_sq.lo - fputil::multiply_add(r_d, r_d, -sum_sq.hi);
#else
    fputil::DoubleDouble r_sq = fputil::exact_mult(r_d, r_d);
    double err = (sum_sq.hi - r_sq.hi) + (sum_sq.lo - r_sq.lo);
#endif

    uint64_t r_u = result.uintval();
    if (err > 0) {
      r_u |= 1;
    } else if ((err < 0) && (r_u & 1) == 0) {
      r_u -= 1;
    }

    return static_cast<float>(DoubleBits(r_u).get_val());
  }

  return static_cast<float>(result.get_val());
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_HYPOTF_H

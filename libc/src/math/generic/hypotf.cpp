//===-- Implementation of hypotf function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/math/hypotf.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, hypotf, (float x, float y)) {
  using DoubleBits = fputil::FPBits<double>;
  using FPBits = fputil::FPBits<float>;

  FPBits x_abs = FPBits(x).abs();
  FPBits y_abs = FPBits(y).abs();

  bool x_abs_larger = x_abs.uintval() >= y_abs.uintval();

  FPBits a_bits = x_abs_larger ? x_abs : y_abs;
  FPBits b_bits = x_abs_larger ? y_abs : x_abs;

  uint32_t a_u = a_bits.uintval();
  uint32_t b_u = b_bits.uintval();

  // Note: replacing `a_u >= FPBits::EXP_MASK` with `a_bits.is_inf_or_nan()`
  // generates extra exponent bit masking instructions on x86-64.
  if (LIBC_UNLIKELY(a_u >= FPBits::EXP_MASK)) {
    // x or y is inf or nan
    if (a_bits.is_signaling_nan() || b_bits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    if (a_bits.is_inf() || b_bits.is_inf())
      return FPBits::inf().get_val();
    return a_bits.get_val();
  }

  if (LIBC_UNLIKELY(a_u - b_u >=
                    static_cast<uint32_t>((FPBits::FRACTION_LEN + 2)
                                          << FPBits::FRACTION_LEN)))
    return x_abs.get_val() + y_abs.get_val();

  double ad = static_cast<double>(a_bits.get_val());
  double bd = static_cast<double>(b_bits.get_val());

  // These squares are exact.
  double a_sq = ad * ad;
#ifdef LIBC_TARGET_CPU_HAS_FMA
  double sum_sq = fputil::multiply_add(bd, bd, a_sq);
#else
  double b_sq = bd * bd;
  double sum_sq = a_sq + b_sq;
#endif

  // Take sqrt in double precision.
  DoubleBits result(fputil::sqrt<double>(sum_sq));
  uint64_t r_u = result.uintval();

  // If any of the sticky bits of the result are non-zero, except the LSB, then
  // the rounded result is correct.
  if (LIBC_UNLIKELY(((r_u + 1) & 0x0000'0000'0FFF'FFFE) == 0)) {
    double r_d = result.get_val();

    // Perform rounding correction.
#ifdef LIBC_TARGET_CPU_HAS_FMA
    double sum_sq_lo = fputil::multiply_add(bd, bd, a_sq - sum_sq);
    double err = sum_sq_lo - fputil::multiply_add(r_d, r_d, -sum_sq);
#else
    fputil::DoubleDouble r_sq = fputil::exact_mult(r_d, r_d);
    double sum_sq_lo = b_sq - (sum_sq - a_sq);
    double err = (sum_sq - r_sq.hi) + (sum_sq_lo - r_sq.lo);
#endif

    if (err > 0) {
      r_u |= 1;
    } else if ((err < 0) && (r_u & 1) == 0) {
      r_u -= 1;
    } else if ((r_u & 0x0000'0000'1FFF'FFFF) == 0) {
      // The rounded result is exact.
      fputil::clear_except_if_required(FE_INEXACT);
    }
    return static_cast<float>(DoubleBits(r_u).get_val());
  }

  return static_cast<float>(result.get_val());
}

} // namespace LIBC_NAMESPACE_DECL

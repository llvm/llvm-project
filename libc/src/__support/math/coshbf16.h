//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the coshbf16 function, which computes the hyperbolic
/// cosine of a bfloat16 value.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_COSHBF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_COSHBF16_H

#include "hdr/fenv_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/except_value_utils.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/sinhcoshbf16_utils.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE bfloat16 coshbf16(bfloat16 x) {
  using FPBits = fputil::FPBits<bfloat16>;
  FPBits x_bits(x);
  // cosh is an even function, so only |x| matters.
  uint16_t x_abs = x_bits.uintval() & 0x7fff;

  // |x| <= 0x1.68p-4 (~0.0879)
  // cosh(x) = 1 + x^2/2 + ..., and for these inputs the excess over 1 is less
  // than half an ulp of 1 in bfloat16 (2^-8), so the result rounds to 1 (or to
  // the next value above 1 when rounding upward, since cosh(x) > 1 for x != 0).
  if (x_abs <= 0x3db4) {
    if (x_abs == 0U)
      return FPBits::one().get_val();

#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
    if (fputil::fenv_is_round_up())
      return FPBits(static_cast<uint16_t>(FPBits::one().uintval() + 1))
          .get_val();
#endif
    return FPBits::one().get_val();
  }

  // |x| >= 89.5 (or inf / NaN)
  if (LIBC_UNLIKELY(x_abs >= 0x42b3)) {
    // cosh(+-inf) = +inf
    if (x_bits.is_inf())
      return FPBits::inf().get_val();

    // cosh(NaN) = NaN
    if (x_bits.is_nan()) {
      if (x_bits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return x;
    }

    // cosh(x) is positive for all x, so the overflow is always to +inf.
    return fputil::set_overflow<bfloat16>();
  }

  // |x| >= 6.875
  // e^(-|x|) is negligible, so cosh(x) ~ e^|x| / 2.
  if (x_abs >= 0x40dc)
    return fputil::cast<bfloat16>(
        math::sinhcoshbf16_internal::exp_half(x_bits.abs().get_val()));

  // cosh(x) = (e^x + e^(-x)) / 2.
  return fputil::cast<bfloat16>(
      math::sinhcoshbf16_internal::eval_sinh_or_cosh</*IsSinh=*/false>(
          static_cast<float>(x)));
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_COSHBF16_H

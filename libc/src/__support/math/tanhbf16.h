//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_TANHBF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_TANHBF16_H

#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "tanhf.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

LIBC_INLINE bfloat16 tanhbf16(bfloat16 x) {
  using FPBits = fputil::FPBits<bfloat16>;
  FPBits x_bits(x);
  uint16_t x_u = x_bits.uintval();
  uint16_t x_abs = x_u & 0x7fffU;

  if (LIBC_UNLIKELY(x_bits.is_nan())) {
    if (x_bits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // tanh(+/-inf) = +/-1, and tanh(+/-0) = +/-0.
  if (LIBC_UNLIKELY(x_bits.is_inf()))
    return FPBits::one(x_bits.sign()).get_val();
  if (LIBC_UNLIKELY(x_abs == 0))
    return x;

  // For |x| <= 0x1.7p-4, tanh(x) rounds to x in round-to-nearest.  For
  // directed rounding, its magnitude is strictly smaller than |x|.
  if (LIBC_UNLIKELY(x_abs <= 0x3db8U)) {
    bool toward_zero = false;
#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
    int rounding_mode = fputil::quick_get_round();
    toward_zero = rounding_mode == FE_TOWARDZERO ||
                  (rounding_mode == FE_DOWNWARD && x_bits.is_pos()) ||
                  (rounding_mode == FE_UPWARD && x_bits.is_neg());
#endif
    uint16_t result = toward_zero ? static_cast<uint16_t>(x_u - 1U) : x_u;
    int exceptions = FE_INEXACT;
    if ((result & 0x7fffU) < 0x0080U)
      exceptions |= FE_UNDERFLOW;
    fputil::raise_except_if_required(exceptions);
    return FPBits(result).get_val();
  }

  // 0x1.bcp+1 is the first bfloat16 input whose tanh rounds to 1 in
  // round-to-nearest.  The result remains below 1 for every finite input.
  if (LIBC_UNLIKELY(x_abs >= 0x405eU)) {
    fputil::raise_except_if_required(FE_INEXACT);
    bool round_to_one = true;
#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
    int rounding_mode = fputil::quick_get_round();
    round_to_one = rounding_mode == FE_TONEAREST ||
                   (rounding_mode == FE_UPWARD && x_bits.is_pos()) ||
                   (rounding_mode == FE_DOWNWARD && x_bits.is_neg());
#endif
    if (round_to_one)
      return FPBits::one(x_bits.sign()).get_val();
    return FPBits(static_cast<uint16_t>((x_u & 0x8000U) | 0x3f7fU)).get_val();
  }

  return fputil::cast<bfloat16>(tanhf(static_cast<float>(x)));
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_TANHBF16_H

//===-- Single-precision atan function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atanf.h"
#include "math_utils.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/generic/inv_trigf_utils.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(float, atanf, (float x)) {
  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);
  bool sign = xbits.get_sign();
  xbits.set_sign(false);

  if (LIBC_UNLIKELY(xbits.is_inf_or_nan())) {
    if (xbits.is_inf())
      return opt_barrier(sign ? -M_MATH_PI_2 : M_MATH_PI_2);
    else
      return x;
  }
  // |x| == 0.06905200332403183
  if (LIBC_UNLIKELY(xbits.uintval() == 0x3d8d6b23U)) {
    if (fputil::get_round() == FE_TONEAREST) {
      // 0.06894256919622421
      FPBits br(0x3d8d31c3U);
      br.set_sign(sign);
      return br.get_val();
    }
  }

  // |x| == 1.8670953512191772
  if (LIBC_UNLIKELY(xbits.uintval() == 0x3feefcfbU)) {
    int rounding_mode = fputil::get_round();
    if (sign) {
      if (rounding_mode == FE_DOWNWARD) {
        // -1.0790828466415405
        return FPBits(0xbf8a1f63U).get_val();
      }
    } else {
      if (rounding_mode == FE_UPWARD) {
        // 1.0790828466415405
        return FPBits(0x3f8a1f63U).get_val();
      }
    }
  }

  return atan_eval(x);
}

} // namespace __llvm_libc

//===-- Half-precision cosh(x) function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/coshf16.h"
#include "expxf16.h"
#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/except_value_utils.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

static constexpr fputil::ExceptValues<float16, 9> COSHF16_EXCEPTS_POS = {{
    // x = 0x1.6ap-5, coshf16(x) = 0x1p+0 (RZ)
    {0x29a8U, 0x3c00U, 1U, 0U, 1U},
    // x = 0x1.8c4p+0, coshf16(x) = 0x1.3a8p+1 (RZ)
    {0x3e31U, 0x40eaU, 1U, 0U, 0U},
    // x = 0x1.994p+0, coshf16(x) = 0x1.498p+1 (RZ)
    {0x3e65U, 0x4126U, 1U, 0U, 0U},
    // x = 0x1.b6p+0, coshf16(x) = 0x1.6d8p+1 (RZ)
    {0x3ed8U, 0x41b6U, 1U, 0U, 1U},
    // x = 0x1.aap+1, coshf16(x) = 0x1.be8p+3 (RZ)
    {0x42a8U, 0x4afaU, 1U, 0U, 1U},
    // x = 0x1.cc4p+1, coshf16(x) = 0x1.23cp+4 (RZ)
    {0x4331U, 0x4c8fU, 1U, 0U, 0U},
    // x = 0x1.288p+2, coshf16(x) = 0x1.9b4p+5 (RZ)
    {0x44a2U, 0x526dU, 1U, 0U, 0U},
    // x = 0x1.958p+2, coshf16(x) = 0x1.1a4p+8 (RZ)
    {0x4656U, 0x5c69U, 1U, 0U, 0U},
    // x = 0x1.5fp+3, coshf16(x) = 0x1.c54p+14 (RZ)
    {0x497cU, 0x7715U, 1U, 0U, 1U},
}};

static constexpr fputil::ExceptValues<float16, 4> COSHF16_EXCEPTS_NEG = {{
    // x = -0x1.6ap-5, coshf16(x) = 0x1p+0 (RZ)
    {0xa9a8U, 0x3c00U, 1U, 0U, 1U},
    // x = -0x1.b6p+0, coshf16(x) = 0x1.6d8p+1 (RZ)
    {0xbed8U, 0x41b6U, 1U, 0U, 1U},
    // x = -0x1.288p+2, coshf16(x) = 0x1.9b4p+5 (RZ)
    {0xc4a2U, 0x526dU, 1U, 0U, 0U},
    // x = -0x1.5fp+3, coshf16(x) = 0x1.c54p+14 (RZ)
    {0xc97cU, 0x7715U, 1U, 0U, 1U},
}};

LLVM_LIBC_FUNCTION(float16, coshf16, (float16 x)) {
  using FPBits = fputil::FPBits<float16>;
  FPBits x_bits(x);

  uint16_t x_u = x_bits.uintval();
  uint16_t x_abs = x_u & 0x7fffU;

  // When |x| >= acosh(2^16), or x is NaN.
  if (LIBC_UNLIKELY(x_abs >= 0x49e5U)) {
    // cosh(NaN) = NaN
    if (x_bits.is_nan()) {
      if (x_bits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }

      return x;
    }

    // When |x| >= acosh(2^16).
    if (x_abs >= 0x49e5U) {
      // cosh(+/-inf) = +inf
      if (x_bits.is_inf())
        return FPBits::inf().get_val();

      switch (fputil::quick_get_round()) {
      case FE_TONEAREST:
      case FE_UPWARD:
        fputil::set_errno_if_required(ERANGE);
        fputil::raise_except_if_required(FE_OVERFLOW | FE_INEXACT);
        return FPBits::inf().get_val();
      default:
        return FPBits::max_normal().get_val();
      }
    }
  }

  if (x_bits.is_pos()) {
    if (auto r = COSHF16_EXCEPTS_POS.lookup(x_u); LIBC_UNLIKELY(r.has_value()))
      return r.value();
  } else {
    if (auto r = COSHF16_EXCEPTS_NEG.lookup(x_u); LIBC_UNLIKELY(r.has_value()))
      return r.value();
  }

  return eval_sinh_or_cosh</*IsSinh=*/false>(x);
}

} // namespace LIBC_NAMESPACE_DECL

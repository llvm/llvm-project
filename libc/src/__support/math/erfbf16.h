//===-- Implementation header for erfbf16 -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ERFBF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ERFBF16_H

#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {
namespace erfbf16_internal {

// Approximate F(z) = erf(sqrt(z)) / sqrt(z)
// on the interval 0 <= z <= 5.
// Coefficients are in descending order.
LIBC_INLINE_VAR constexpr double COEFFS[12] = {
    -0x1.30d6be1481948p-33, 0x1.7cf774adce86ep-28,  -0x1.d1591ee4c2994p-24,
    0x1.7c231d88f04b8p-20,  -0x1.de87683f1fc4bp-17, 0x1.f41cbbaa3fd0bp-14,
    -0x1.bf42adf1e3d24p-11, 0x1.5642bc032f199p-8,   -0x1.b829b41499428p-6,
    0x1.ce2eebc4dd9eep-4,   -0x1.81274549391d9p-2,  0x1.20dd75029b655p+0,
};

} // namespace erfbf16_internal

LIBC_INLINE bfloat16 erfbf16(bfloat16 x) {
  using FPBits = fputil::FPBits<bfloat16>;

  const FPBits x_bits(x);
  const uint16_t x_abs = x_bits.uintval() & 0x7fffU;
  const bool negative = x_bits.is_neg();

  // Handle NaN.
  if (LIBC_UNLIKELY(x_bits.is_nan())) {
    if (x_bits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // erf(+/-inf) = +/-1.
  if (LIBC_UNLIKELY(x_bits.is_inf()))
    return FPBits(static_cast<uint16_t>(negative ? 0xbf80U : 0x3f80U))
        .get_val();

  // Preserve signed zero.
  if (LIBC_UNLIKELY(x_abs == 0))
    return x;

  // Handle large finite values.
  if (LIBC_UNLIKELY(x_abs >= 0x400dU)) {
    int mode = fputil::get_round();

    const uint16_t result_bits =
        negative
            ? ((mode == FE_UPWARD || mode == FE_TOWARDZERO) ? 0xbf7fU : 0xbf80U)
            : ((mode == FE_DOWNWARD || mode == FE_TOWARDZERO) ? 0x3f7fU
                                                              : 0x3f80U);

    fputil::raise_except_if_required(FE_INEXACT);
    return FPBits(result_bits).get_val();
  }

  // Evaluate erf(x) using a polynomial approximation.
  const double ax = fputil::cast<double>(FPBits(x_abs).get_val());
  const double z = ax * ax;

  double p = erfbf16_internal::COEFFS[0];

  for (int i = 1; i < 12; ++i)
    p = fputil::multiply_add(p, z, erfbf16_internal::COEFFS[i]);

  const double y = negative ? -(ax * p) : (ax * p);

  const bfloat16 rounded = fputil::cast<bfloat16>(y);

  fputil::raise_except_if_required(FE_INEXACT);

  return rounded;
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ERFBF16_H

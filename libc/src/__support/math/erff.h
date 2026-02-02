//===-- Implementation header for erff --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ERFF_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ERFF_H

#include "common_constants.h" // ERFF_COEFFS
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/except_value_utils.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE static constexpr float erff(float x) {
  using namespace common_constants_internal;

  using FPBits = typename fputil::FPBits<float>;
  FPBits xbits(x);

  uint32_t x_u = xbits.uintval();
  uint32_t x_abs = x_u & 0x7fff'ffffU;

  if (LIBC_UNLIKELY(x_abs >= 0x4080'0000U)) {
    constexpr float ONE[2] = {1.0f, -1.0f};
    constexpr float SMALL[2] = {-0x1.0p-25f, 0x1.0p-25f};

    int sign = xbits.is_neg() ? 1 : 0;

    if (LIBC_UNLIKELY(x_abs >= 0x7f80'0000U)) {
      if (xbits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return (x_abs > 0x7f80'0000) ? x : ONE[sign];
    }

    return ONE[sign] + SMALL[sign];
  }

#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
  // Exceptional mask = common 0 bits of 2 exceptional values.
  constexpr uint32_t EXCEPT_MASK = 0x809a'6184U;

  if (LIBC_UNLIKELY((x_abs & EXCEPT_MASK) == 0)) {
    // Exceptional values
    if (LIBC_UNLIKELY(x_abs == 0x3f65'9229U)) // |x| = 0x1.cb2452p-1f
      return x < 0.0f ? fputil::round_result_slightly_down(-0x1.972ea8p-1f)
                      : fputil::round_result_slightly_up(0x1.972ea8p-1f);
    if (LIBC_UNLIKELY(x_abs == 0x4004'1e6aU)) // |x| = 0x1.083cd4p+1f
      return x < 0.0f ? fputil::round_result_slightly_down(-0x1.fe3462p-1f)
                      : fputil::round_result_slightly_up(0x1.fe3462p-1f);
    if (x_abs == 0U)
      return x;
  }
#endif // !LIBC_MATH_HAS_SKIP_ACCURATE_PASS

  // Polynomial approximation:
  //   erf(x) ~ x * (c0 + c1 * x^2 + c2 * x^4 + ... + c7 * x^14)
  double xd = static_cast<double>(x);
  double xsq = xd * xd;

  constexpr uint32_t EIGHT = 3 << FPBits::FRACTION_LEN;
  int idx = static_cast<int>(FPBits(x_abs + EIGHT).get_val());

  double x4 = xsq * xsq;
  double c0 =
      fputil::multiply_add(xsq, ERFF_COEFFS[idx][1], ERFF_COEFFS[idx][0]);
  double c1 =
      fputil::multiply_add(xsq, ERFF_COEFFS[idx][3], ERFF_COEFFS[idx][2]);
  double c2 =
      fputil::multiply_add(xsq, ERFF_COEFFS[idx][5], ERFF_COEFFS[idx][4]);
  double c3 =
      fputil::multiply_add(xsq, ERFF_COEFFS[idx][7], ERFF_COEFFS[idx][6]);

  double x8 = x4 * x4;
  double p0 = fputil::multiply_add(x4, c1, c0);
  double p1 = fputil::multiply_add(x4, c3, c2);

  return static_cast<float>(xd * fputil::multiply_add(x8, p1, p0));
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ERFF_H

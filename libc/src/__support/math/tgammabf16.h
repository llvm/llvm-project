//===-- Implementation header for tgammabf16 --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_TGAMMABF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_TGAMMABF16_H

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/exp.h"
#include "src/__support/math/log.h"
#include "src/__support/math/sin.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

namespace tgammabf16_internal {

LIBC_INLINE_VAR constexpr double PI = 0x1.921fb54442d18p+1;
LIBC_INLINE_VAR constexpr double LOG_SQRT_2_PI = 0x1.d67f1c864beb5p-1;

// Paul Godfrey's exact Lanczos approximation coefficients (g=7, n=9)
LIBC_INLINE_VAR constexpr double LANCZOS_COEFFS[9] = {
    0.99999999999980993,  676.5203681218851,     -1259.1392167224028,
    771.32342877765313,   -176.61502916214059,   12.507343278224757,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};

LIBC_INLINE bool is_negative_integer(double x) {
  if (x > -1.0)
    return false;
  if (x <= -128.0)
    return true;
  int n = static_cast<int>(x);
  return x == static_cast<double>(n);
}

} // namespace tgammabf16_internal

LIBC_INLINE bfloat16 tgammabf16(bfloat16 x) {
  using FPBits = fputil::FPBits<bfloat16>;
  using namespace tgammabf16_internal;

  FPBits xbits(x);

  if (LIBC_UNLIKELY(xbits.is_nan())) {
    if (xbits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  if (LIBC_UNLIKELY(xbits.is_inf())) {
    if (xbits.is_pos())
      return x;
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  if (LIBC_UNLIKELY(xbits.is_zero())) {
    fputil::set_errno_if_required(ERANGE);
    fputil::raise_except_if_required(FE_DIVBYZERO);
    return FPBits::inf(xbits.sign()).get_val();
  }

  double xd = static_cast<double>(static_cast<float>(x));
  if (LIBC_UNLIKELY(is_negative_integer(xd))) {
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  // Fast path for exact positive integers
  if (xd > 0.0 && xd <= 35.0) {
    int n = static_cast<int>(xd);
    if (xd == static_cast<double>(n)) {
      double res = 1.0;
      for (int i = 1; i < n; ++i)
        res *= i;
      return fputil::cast<bfloat16>(res);
    }
  }

  bool reflection = false;
  bool divide_by_x = false;
  double x_eval = xd;
  double res;

  // Fast path for tiny positive inputs to prevent exact-boundary overshoots
  // For tiny x, Gamma(x) ~= 1/x - gamma
  if (LIBC_UNLIKELY(xd > 0.0 && xd < 0x1.0p-8)) {
    res = (1.0 / xd) - 0.577215664901532860606;
  } else {
    if (xd < 0.0) {
      reflection = true;
      x_eval = 1.0 - xd;
    } else if (xd < 1.0) {
      divide_by_x = true;
      x_eval = xd + 1.0;
    }

    double z = x_eval - 1.0;
    double a = LANCZOS_COEFFS[0];
    for (int i = 1; i < 9; ++i) {
      a += LANCZOS_COEFFS[i] / (z + static_cast<double>(i));
    }

    double t = z + 7.5;

    // ILP Optimization: FMA for the exponent argument
    double log_t = math::log(t);
    double base_term = (LOG_SQRT_2_PI - t) + math::log(a);
    double exp_arg = fputil::multiply_add(z + 0.5, log_t, base_term);

    res = math::exp(exp_arg);

    if (reflection) {
      double sin_pi_x = math::sin(PI * xd);
      double denom = sin_pi_x * res;

      if (LIBC_UNLIKELY(denom == 0.0)) {
        fputil::set_errno_if_required(ERANGE);
        fputil::raise_except_if_required(FE_UNDERFLOW);
        return FPBits::zero(Sign::POS).get_val();
      }
      res = PI / denom;
    } else if (divide_by_x) {
      res = res / xd;
    }
  }

  double abs_res = res < 0.0 ? -res : res;

  // 0x1.ffp127 is max normal + 0.5 ULP in bfloat16, the exact RTN overflow
  // threshold
  if (LIBC_UNLIKELY(abs_res > 0x1.ffp+127)) {
    fputil::set_errno_if_required(ERANGE);
    fputil::raise_except_if_required(FE_OVERFLOW);

#ifdef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
    return FPBits::inf(res > 0.0 ? Sign::POS : Sign::NEG).get_val();
#else
    Sign sign = res > 0.0 ? Sign::POS : Sign::NEG;
    switch (fputil::quick_get_round()) {
    case FE_TONEAREST:
      return FPBits::inf(sign).get_val();
    case FE_UPWARD:
      return sign == Sign::POS ? FPBits::inf(Sign::POS).get_val()
                               : FPBits::max_normal(Sign::NEG).get_val();
    case FE_DOWNWARD:
      return sign == Sign::POS ? FPBits::max_normal(Sign::POS).get_val()
                               : FPBits::inf(Sign::NEG).get_val();
    case FE_TOWARDZERO:
      return FPBits::max_normal(sign).get_val();
    default:
      return FPBits::max_normal(sign).get_val();
    }
#endif
  }

  return fputil::cast<bfloat16>(res);
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_TGAMMABF16_H

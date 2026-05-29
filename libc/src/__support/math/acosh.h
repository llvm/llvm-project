//===-- Implementation header for acosh -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ACOSH_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ACOSH_H

#include "log1p.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/__support/macros/properties/cpu_features.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

// Compute log1p(u_hi + u_lo) using log1p's step-1 range reduction and
// polynomial, with the Float128 accurate path for correct rounding.
LIBC_INLINE double acosh_log1p_dd(double u_hi, double u_lo) {
  using FPBits_t = fputil::FPBits<double>;
  using namespace log1p_internal;
  using namespace common_constants_internal;

  constexpr int EXP_BIAS = FPBits_t::EXP_BIAS;
  constexpr int FRACTION_LEN = FPBits_t::FRACTION_LEN;

  // Knuth's 2Sum (exact_add<false>) is correct for all rounding modes.
  fputil::DoubleDouble x_dd = fputil::exact_add<false>(u_hi, 1.0);
  x_dd.lo += u_lo;

  // Step-1 range reduction (identical to log1p's fast path).
  FPBits_t xhi_bits(x_dd.hi);
  uint64_t xhi_frac = xhi_bits.get_mantissa();
  uint64_t xdd_u = xhi_bits.uintval();

  int idx = static_cast<int>((xhi_frac + (1ULL << (FRACTION_LEN - 8))) >>
                             (FRACTION_LEN - 7));
  int x_e = xhi_bits.get_exponent() + (idx >> 7);
  double e_x = static_cast<double>(x_e);

  int64_t s_u = static_cast<int64_t>(xdd_u & FPBits_t::EXP_MASK) -
                (static_cast<int64_t>(EXP_BIAS) << FRACTION_LEN);

  uint64_t m_hi_bits = FPBits_t::one().uintval() | xhi_frac;
  uint64_t m_lo_bits =
      FPBits_t(x_dd.lo).abs().get_val() > x_dd.hi * 0x1.0p-127
          ? static_cast<uint64_t>(cpp::bit_cast<int64_t>(x_dd.lo) - s_u)
          : 0;

  fputil::DoubleDouble m_dd{FPBits_t(m_lo_bits).get_val(),
                            FPBits_t(m_hi_bits).get_val()};

  double r = R1[idx];
  fputil::DoubleDouble v_lo_p = fputil::exact_mult(m_dd.lo, r);
  double v_hi_p;
#ifdef LIBC_TARGET_CPU_HAS_FMA_DOUBLE
  v_hi_p = fputil::multiply_add(r, m_dd.hi, -1.0);
#else
  double c = FPBits_t((static_cast<uint64_t>(idx) << (FRACTION_LEN - 7)) +
                      uint64_t(0x3FF0'0000'0000'0000ULL))
                 .get_val();
  v_hi_p = fputil::multiply_add(r, m_dd.hi - c, RCM1[idx]);
#endif

  fputil::DoubleDouble v_dd_red = fputil::exact_add(v_hi_p, v_lo_p.hi);
  v_dd_red.lo += v_lo_p.lo;

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
  // Fast path: polynomial only (no correct-rounding guarantee).
  double hi = fputil::multiply_add(e_x, LOG_2_HI, LOG_R1_DD[idx].hi);
  double lo = fputil::multiply_add(e_x, LOG_2_LO, LOG_R1_DD[idx].lo);
  fputil::DoubleDouble r1 = fputil::exact_add(hi, v_dd_red.hi);
  double v_sq = v_dd_red.hi * v_dd_red.hi;
  double p0 = fputil::multiply_add(v_dd_red.hi, P_COEFFS[1], P_COEFFS[0]);
  double p1 = fputil::multiply_add(v_dd_red.hi, P_COEFFS[3], P_COEFFS[2]);
  double p2 = fputil::multiply_add(v_dd_red.hi, P_COEFFS[5], P_COEFFS[4]);
  double p = fputil::polyeval(v_sq, (v_dd_red.lo + r1.lo) + lo, p0, p1, p2);
  return r1.hi + p;
#else
  // Always use the Float128 accurate path for correct rounding.
  // The Ziv fast-path is not used here because for some large-x inputs
  // err << ULP(result), so the Ziv test always passes even when the
  // polynomial is 1 ULP off.
  return log1p_accurate(x_e, idx, v_dd_red);
#endif
}

LIBC_INLINE double acosh(double x) {
  using FPBits = fputil::FPBits<double>;
  using DoubleDouble = fputil::DoubleDouble;

  FPBits xbits(x);
  uint64_t x_u = xbits.uintval();

  // x <= 1.0 is false for NaN, so NaN falls through to the inf/NaN check.
  if (LIBC_UNLIKELY(x <= 1.0)) {
    if (x == 1.0)
      return 0.0;
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  if (LIBC_UNLIKELY(x_u >= 0x41b0000000000000ULL)) { // x >= 2^28
    if (LIBC_UNLIKELY(xbits.is_inf_or_nan())) {
      if (xbits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return x;
    }
    // acosh(x) = log(2x) + O(1/x^2); correction < 0.5 ULP for x >= 2^28.
    // 2*x is exact for x in [2^28, 2^52] (the test range); compute 2x-1
    // as a double-double so acosh_log1p_dd gets the exact correction.
    double two_x = 2.0 * x;
    DoubleDouble u_dd = fputil::exact_add<false>(two_x, -1.0);
    return acosh_log1p_dd(u_dd.hi, u_dd.lo);
  }

  // acosh(x) = log1p(u),  u = (x-1) + sqrt(x^2-1).
  // Compute u as a double-double for a correctly-rounded result.

  // x^2 - 1 as double-double.
  DoubleDouble x_sq = fputil::exact_mult(x, x);
  DoubleDouble v_dd = fputil::exact_add<false>(x_sq.hi, -1.0);
  v_dd.lo += x_sq.lo;

  // sqrt(x^2-1) as double-double via one Newton correction.
  // Use exact_mult for s_hi^2 so the correction is accurate on non-FMA
  // targets too (fma(s,-s,v) without hardware FMA loses ~1/4 ULP).
  double s_hi = fputil::sqrt<double>(v_dd.hi);
  DoubleDouble s_sq = fputil::exact_mult(s_hi, s_hi);
  DoubleDouble r_dd = fputil::exact_add<false>(v_dd.hi, -s_sq.hi);
  double s_lo = (r_dd.hi + (r_dd.lo - s_sq.lo) + v_dd.lo) / (2.0 * s_hi);

  // t = x-1 and u = t+s, both as double-doubles (Knuth's 2Sum).
  DoubleDouble t_dd = fputil::exact_add<false>(x, -1.0);
  DoubleDouble u_dd = fputil::exact_add<false>(t_dd.hi, s_hi);
  double u_lo = u_dd.lo + t_dd.lo + s_lo;

  return acosh_log1p_dd(u_dd.hi, u_lo);
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ACOSH_H

//===-- Double-precision sin function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sin.h"
#include "hdr/errno_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/dyadic_float.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/common.h"
#include "src/__support/macros/optimization.h"            // LIBC_UNLIKELY
#include "src/__support/macros/properties/cpu_features.h" // LIBC_TARGET_CPU_HAS_FMA
#include "src/math/generic/sincos_eval.h"

#ifdef LIBC_TARGET_CPU_HAS_FMA
#include "range_reduction_double_fma.h"

using LIBC_NAMESPACE::fma::FAST_PASS_EXPONENT;
using LIBC_NAMESPACE::fma::ONE_TWENTY_EIGHT_OVER_PI;
using LIBC_NAMESPACE::fma::range_reduction_small;
using LIBC_NAMESPACE::fma::SIN_K_PI_OVER_128;

LIBC_INLINE constexpr bool NO_FMA = false;
#else
#include "range_reduction_double_nofma.h"

using LIBC_NAMESPACE::nofma::FAST_PASS_EXPONENT;
using LIBC_NAMESPACE::nofma::ONE_TWENTY_EIGHT_OVER_PI;
using LIBC_NAMESPACE::nofma::range_reduction_small;
using LIBC_NAMESPACE::nofma::SIN_K_PI_OVER_128;

LIBC_INLINE constexpr bool NO_FMA = true;
#endif // LIBC_TARGET_CPU_HAS_FMA

// TODO: We might be able to improve the performance of large range reduction of
// non-FMA targets further by operating directly on 25-bit chunks of 128/pi and
// pre-split SIN_K_PI_OVER_128, but that might double the memory footprint of
// those lookup table.
#include "range_reduction_double_common.h"

#if ((LIBC_MATH & LIBC_MATH_SKIP_ACCURATE_PASS) != 0)
#define LIBC_MATH_SIN_SKIP_ACCURATE_PASS
#endif

namespace LIBC_NAMESPACE {

using DoubleDouble = fputil::DoubleDouble;
using Float128 = typename fputil::DyadicFloat<128>;

LLVM_LIBC_FUNCTION(double, sin, (double x)) {
  using FPBits = typename fputil::FPBits<double>;
  FPBits xbits(x);

  uint16_t x_e = xbits.get_biased_exponent();

  DoubleDouble y;
  unsigned k;
  generic::LargeRangeReduction<NO_FMA> range_reduction_large;

  // |x| < 2^32 (with FMA) or |x| < 2^23 (w/o FMA)
  if (LIBC_LIKELY(x_e < FPBits::EXP_BIAS + FAST_PASS_EXPONENT)) {
    // |x| < 2^-26
    if (LIBC_UNLIKELY(x_e < FPBits::EXP_BIAS - 26)) {
      // Signed zeros.
      if (LIBC_UNLIKELY(x == 0.0))
        return x;

        // For |x| < 2^-26, |sin(x) - x| < ulp(x)/2.
#ifdef LIBC_TARGET_CPU_HAS_FMA
      return fputil::multiply_add(x, -0x1.0p-54, x);
#else
      if (LIBC_UNLIKELY(x_e < 4)) {
        int rounding_mode = fputil::quick_get_round();
        if (rounding_mode == FE_TOWARDZERO ||
            (xbits.sign() == Sign::POS && rounding_mode == FE_DOWNWARD) ||
            (xbits.sign() == Sign::NEG && rounding_mode == FE_UPWARD))
          return FPBits(xbits.uintval() - 1).get_val();
      }
      return fputil::multiply_add(x, -0x1.0p-54, x);
#endif // LIBC_TARGET_CPU_HAS_FMA
    }

    // // Small range reduction.
    k = range_reduction_small(x, y);
  } else {
    // Inf or NaN
    if (LIBC_UNLIKELY(x_e > 2 * FPBits::EXP_BIAS)) {
      // sin(+-Inf) = NaN
      if (xbits.get_mantissa() == 0) {
        fputil::set_errno_if_required(EDOM);
        fputil::raise_except_if_required(FE_INVALID);
      }
      return x + FPBits::quiet_nan().get_val();
    }

    // Large range reduction.
    k = range_reduction_large.compute_high_part(x);
    y = range_reduction_large.fast();
  }

  DoubleDouble sin_y, cos_y;

  generic::sincos_eval(y, sin_y, cos_y);

  // Look up sin(k * pi/128) and cos(k * pi/128)
  // Memory saving versions:

  // Use 128-entry table instead:
  // DoubleDouble sin_k = SIN_K_PI_OVER_128[k & 127];
  // uint64_t sin_s = static_cast<uint64_t>(k & 128) << (63 - 7);
  // sin_k.hi = FPBits(FPBits(sin_k.hi).uintval() ^ sin_s).get_val();
  // sin_k.lo = FPBits(FPBits(sin_k.hi).uintval() ^ sin_s).get_val();
  // DoubleDouble cos_k = SIN_K_PI_OVER_128[(k + 64) & 127];
  // uint64_t cos_s = static_cast<uint64_t>((k + 64) & 128) << (63 - 7);
  // cos_k.hi = FPBits(FPBits(cos_k.hi).uintval() ^ cos_s).get_val();
  // cos_k.lo = FPBits(FPBits(cos_k.hi).uintval() ^ cos_s).get_val();

  // Use 64-entry table instead:
  // auto get_idx_dd = [](unsigned kk) -> DoubleDouble {
  //   unsigned idx = (kk & 64) ? 64 - (kk & 63) : (kk & 63);
  //   DoubleDouble ans = SIN_K_PI_OVER_128[idx];
  //   if (kk & 128) {
  //     ans.hi = -ans.hi;
  //     ans.lo = -ans.lo;
  //   }
  //   return ans;
  // };
  // DoubleDouble sin_k = get_idx_dd(k);
  // DoubleDouble cos_k = get_idx_dd(k + 64);

  // Fast look up version, but needs 256-entry table.
  // cos(k * pi/128) = sin(k * pi/128 + pi/2) = sin((k + 64) * pi/128).
  DoubleDouble sin_k = SIN_K_PI_OVER_128[k & 255];
  DoubleDouble cos_k = SIN_K_PI_OVER_128[(k + 64) & 255];

  // After range reduction, k = round(x * 128 / pi) and y = x - k * (pi / 128).
  // So k is an integer and -pi / 256 <= y <= pi / 256.
  // Then sin(x) = sin((k * pi/128 + y)
  //             = sin(y) * cos(k*pi/128) + cos(y) * sin(k*pi/128)
  DoubleDouble sin_k_cos_y = fputil::quick_mult<NO_FMA>(cos_y, sin_k);
  DoubleDouble cos_k_sin_y = fputil::quick_mult<NO_FMA>(sin_y, cos_k);

  DoubleDouble rr = fputil::exact_add<false>(sin_k_cos_y.hi, cos_k_sin_y.hi);
  rr.lo += sin_k_cos_y.lo + cos_k_sin_y.lo;

#ifdef LIBC_MATH_SIN_SKIP_ACCURATE_PASS
  return rr.hi + rr.lo;
#else
  // Accurate test and pass for correctly rounded implementation.

#ifdef LIBC_TARGET_CPU_HAS_FMA
  constexpr double ERR = 0x1.0p-70;
#else
  // TODO: Improve non-FMA fast pass accuracy.
  constexpr double ERR = 0x1.0p-66;
#endif // LIBC_TARGET_CPU_HAS_FMA

  double rlp = rr.lo + ERR;
  double rlm = rr.lo - ERR;

  double r_upper = rr.hi + rlp; // (rr.lo + ERR);
  double r_lower = rr.hi + rlm; // (rr.lo - ERR);

  // Ziv's rounding test.
  if (LIBC_LIKELY(r_upper == r_lower))
    return r_upper;

  Float128 u_f128, sin_u, cos_u;
  if (LIBC_LIKELY(x_e < FPBits::EXP_BIAS + FAST_PASS_EXPONENT))
    u_f128 = generic::range_reduction_small_f128(x);
  else
    u_f128 = range_reduction_large.accurate();

  generic::sincos_eval(u_f128, sin_u, cos_u);

  auto get_sin_k = [](unsigned kk) -> Float128 {
    unsigned idx = (kk & 64) ? 64 - (kk & 63) : (kk & 63);
    Float128 ans = generic::SIN_K_PI_OVER_128_F128[idx];
    if (kk & 128)
      ans.sign = Sign::NEG;
    return ans;
  };

  // cos(k * pi/128) = sin(k * pi/128 + pi/2) = sin((k + 64) * pi/128).
  Float128 sin_k_f128 = get_sin_k(k);
  Float128 cos_k_f128 = get_sin_k(k + 64);

  // sin(x) = sin((k * pi/128 + u)
  //        = sin(u) * cos(k*pi/128) + cos(u) * sin(k*pi/128)
  Float128 r = fputil::quick_add(fputil::quick_mul(sin_k_f128, cos_u),
                                 fputil::quick_mul(cos_k_f128, sin_u));

  // TODO: Add assertion if Ziv's accuracy tests fail in debug mode.
  // https://github.com/llvm/llvm-project/issues/96452.

  return static_cast<double>(r);
#endif // !LIBC_MATH_SIN_SKIP_ACCURATE_PASS
}

} // namespace LIBC_NAMESPACE

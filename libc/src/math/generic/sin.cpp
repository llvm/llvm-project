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
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/dyadic_float.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/nearest_integer.h"
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

namespace {

#ifndef LIBC_MATH_SIN_SKIP_ACCURATE_PASS
LIBC_INLINE constexpr Float128 SIN_K_PI_OVER_128_F128[65] = {
    {Sign::POS, 0, 0},
    {Sign::POS, -133, 0xc90a'afbd'1b33'efc9'c539'edcb'fda0'cf2c_u128},
    {Sign::POS, -132, 0xc8fb'2f88'6ec0'9f37'6a17'954b'2b7c'5171_u128},
    {Sign::POS, -131, 0x96a9'0496'70cf'ae65'f775'7409'4d3c'35c4_u128},
    {Sign::POS, -131, 0xc8bd'35e1'4da1'5f0e'c739'6c89'4bbf'7389_u128},
    {Sign::POS, -131, 0xfab2'72b5'4b98'71a2'7047'29ae'56d7'8a37_u128},
    {Sign::POS, -130, 0x9640'8374'7309'd113'000a'89a1'1e07'c1fe_u128},
    {Sign::POS, -130, 0xaf10'a224'59fe'32a6'3fee'f3bb'58b1'f10d_u128},
    {Sign::POS, -130, 0xc7c5'c1e3'4d30'55b2'5cc8'c00e'4fcc'd850_u128},
    {Sign::POS, -130, 0xe05c'1353'f27b'17e5'0ebc'61ad'e6ca'83cd_u128},
    {Sign::POS, -130, 0xf8cf'cbd9'0af8'd57a'4221'dc4b'a772'598d_u128},
    {Sign::POS, -129, 0x888e'9315'8fb3'bb04'9841'56f5'5334'4306_u128},
    {Sign::POS, -129, 0x94a0'3176'acf8'2d45'ae4b'a773'da6b'f754_u128},
    {Sign::POS, -129, 0xa09a'e4a0'bb30'0a19'2f89'5f44'a303'cc0b_u128},
    {Sign::POS, -129, 0xac7c'd3ad'58fe'e7f0'811f'9539'84ef'f83e_u128},
    {Sign::POS, -129, 0xb844'2987'd22c'f576'9cc3'ef36'746d'e3b8_u128},
    {Sign::POS, -129, 0xc3ef'1535'754b'168d'3122'c2a5'9efd'dc37_u128},
    {Sign::POS, -129, 0xcf7b'ca1d'476c'516d'a812'90bd'baad'62e4_u128},
    {Sign::POS, -129, 0xdae8'804f'0ae6'015b'362c'b974'182e'3030_u128},
    {Sign::POS, -129, 0xe633'74c9'8e22'f0b4'2872'ce1b'fc7a'd1cd_u128},
    {Sign::POS, -129, 0xf15a'e9c0'37b1'd8f0'6c48'e9e3'420b'0f1e_u128},
    {Sign::POS, -129, 0xfc5d'26df'c4d5'cfda'27c0'7c91'1290'b8d1_u128},
    {Sign::POS, -128, 0x839c'3cc9'17ff'6cb4'bfd7'9717'f288'0abf_u128},
    {Sign::POS, -128, 0x88f5'9aa0'da59'1421'b892'ca83'61d8'c84c_u128},
    {Sign::POS, -128, 0x8e39'd9cd'7346'4364'bba4'cfec'bff5'4867_u128},
    {Sign::POS, -128, 0x9368'2a66'e896'f544'b178'2191'1e71'c16e_u128},
    {Sign::POS, -128, 0x987f'bfe7'0b81'a708'19ce'c845'ac87'a5c6_u128},
    {Sign::POS, -128, 0x9d7f'd149'0285'c9e3'e25e'3954'9638'ae68_u128},
    {Sign::POS, -128, 0xa267'9928'48ee'b0c0'3b51'67ee'359a'234e_u128},
    {Sign::POS, -128, 0xa736'55df'1f2f'489e'149f'6e75'9934'68a3_u128},
    {Sign::POS, -128, 0xabeb'49a4'6764'fd15'1bec'da80'89c1'a94c_u128},
    {Sign::POS, -128, 0xb085'baa8'e966'f6da'e4ca'd00d'5c94'bcd2_u128},
    {Sign::POS, -128, 0xb504'f333'f9de'6484'597d'89b3'754a'be9f_u128},
    {Sign::POS, -128, 0xb968'41bf'7ffc'b21a'9de1'e3b2'2b8b'f4db_u128},
    {Sign::POS, -128, 0xbdae'f913'557d'76f0'ac85'320f'528d'6d5d_u128},
    {Sign::POS, -128, 0xc1d8'705f'fcbb'6e90'bdf0'715c'b8b2'0bd7_u128},
    {Sign::POS, -128, 0xc5e4'0358'a8ba'05a7'43da'25d9'9267'326b_u128},
    {Sign::POS, -128, 0xc9d1'124c'931f'da7a'8335'241b'e169'3225_u128},
    {Sign::POS, -128, 0xcd9f'023f'9c3a'059e'23af'31db'7179'a4aa_u128},
    {Sign::POS, -128, 0xd14d'3d02'313c'0eed'744f'ea20'e8ab'ef92_u128},
    {Sign::POS, -128, 0xd4db'3148'750d'1819'f630'e8b6'dac8'3e69_u128},
    {Sign::POS, -128, 0xd848'52c0'a80f'fcdb'24b9'fe00'6635'74a4_u128},
    {Sign::POS, -128, 0xdb94'1a28'cb71'ec87'2c19'b632'53da'43fc_u128},
    {Sign::POS, -128, 0xdebe'0563'7ca9'4cfb'4b19'aa71'fec3'ae6d_u128},
    {Sign::POS, -128, 0xe1c5'978c'05ed'8691'f4e8'a837'2f8c'5810_u128},
    {Sign::POS, -128, 0xe4aa'5909'a08f'a7b4'1227'85ae'67f5'515d_u128},
    {Sign::POS, -128, 0xe76b'd7a1'e63b'9786'1251'2952'9d48'a92f_u128},
    {Sign::POS, -128, 0xea09'a68a'6e49'cd62'15ad'45b4'a1b5'e823_u128},
    {Sign::POS, -128, 0xec83'5e79'946a'3145'7e61'0231'ac1d'6181_u128},
    {Sign::POS, -128, 0xeed8'9db6'6611'e307'86f8'c20f'b664'b01b_u128},
    {Sign::POS, -128, 0xf109'0827'b437'25fd'6712'7db3'5b28'7316_u128},
    {Sign::POS, -128, 0xf314'4762'4708'8f74'a548'6bdc'455d'56a2_u128},
    {Sign::POS, -128, 0xf4fa'0ab6'316e'd2ec'163c'5c7f'03b7'18c5_u128},
    {Sign::POS, -128, 0xf6ba'073b'424b'19e8'2c79'1f59'cc1f'fc23_u128},
    {Sign::POS, -128, 0xf853'f7dc'9186'b952'c7ad'c6b4'9888'91bb_u128},
    {Sign::POS, -128, 0xf9c7'9d63'272c'4628'4504'ae08'd19b'2980_u128},
    {Sign::POS, -128, 0xfb14'be7f'bae5'8156'2172'a361'fd2a'722f_u128},
    {Sign::POS, -128, 0xfc3b'27d3'8a5d'49ab'2567'78ff'cb5c'1769_u128},
    {Sign::POS, -128, 0xfd3a'abf8'4528'b50b'eae6'bd95'1c1d'abbe_u128},
    {Sign::POS, -128, 0xfe13'2387'0cfe'9a3d'90cd'1d95'9db6'74ef_u128},
    {Sign::POS, -128, 0xfec4'6d1e'8929'2cf0'4139'0efd'c726'e9ef_u128},
    {Sign::POS, -128, 0xff4e'6d68'0c41'd0a9'0f66'8633'f1ab'858a_u128},
    {Sign::POS, -128, 0xffb1'0f1b'cb6b'ef1d'421e'8eda'af59'453e_u128},
    {Sign::POS, -128, 0xffec'4304'2668'65d9'5657'5523'6696'1732_u128},
    {Sign::POS, 0, 1},
};

#ifdef LIBC_TARGET_CPU_HAS_FMA
constexpr double ERR = 0x1.0p-70;
#else
// TODO: Improve non-FMA fast pass accuracy.
constexpr double ERR = 0x1.0p-66;
#endif // LIBC_TARGET_CPU_HAS_FMA

#endif // !LIBC_MATH_SIN_SKIP_ACCURATE_PASS

} // anonymous namespace

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

  sincos_eval(y, sin_y, cos_y);

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

  FPBits sk_cy(sin_k_cos_y.hi);
  FPBits ck_sy(cos_k_sin_y.hi);
  DoubleDouble rr = fputil::exact_add<false>(sin_k_cos_y.hi, cos_k_sin_y.hi);
  rr.lo += sin_k_cos_y.lo + cos_k_sin_y.lo;

#ifdef LIBC_MATH_SIN_SKIP_ACCURATE_PASS
  return rr.hi + rr.lo;
#else
  // Accurate test and pass for correctly rounded implementation.
  double rlp = rr.lo + ERR;
  double rlm = rr.lo - ERR;

  double r_upper = rr.hi + rlp; // (rr.lo + ERR);
  double r_lower = rr.hi + rlm; // (rr.lo - ERR);

  // Ziv's rounding test.
  if (LIBC_LIKELY(r_upper == r_lower))
    return r_upper;

  Float128 u_f128;
  if (LIBC_LIKELY(x_e < FPBits::EXP_BIAS + FAST_PASS_EXPONENT))
    u_f128 = generic::range_reduction_small_f128(x);
  else
    u_f128 = range_reduction_large.accurate();

  Float128 u_sq = fputil::quick_mul(u_f128, u_f128);

  // sin(u) ~ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11! + x^13/13!
  constexpr Float128 SIN_COEFFS[] = {
      {Sign::POS, -127, 0x80000000'00000000'00000000'00000000_u128}, // 1
      {Sign::NEG, -130, 0xaaaaaaaa'aaaaaaaa'aaaaaaaa'aaaaaaab_u128}, // -1/3!
      {Sign::POS, -134, 0x88888888'88888888'88888888'88888889_u128}, // 1/5!
      {Sign::NEG, -140, 0xd00d00d0'0d00d00d'00d00d00'd00d00d0_u128}, // -1/7!
      {Sign::POS, -146, 0xb8ef1d2a'b6399c7d'560e4472'800b8ef2_u128}, // 1/9!
      {Sign::NEG, -153, 0xd7322b3f'aa271c7f'3a3f25c1'bee38f10_u128}, // -1/11!
      {Sign::POS, -160, 0xb092309d'43684be5'1c198e91'd7b4269e_u128}, // 1/13!
  };

  // cos(u) ~ 1 - x^2/2 + x^4/4! - x^6/6! + x^8/8! - x^10/10! + x^12/12!
  constexpr Float128 COS_COEFFS[] = {
      {Sign::POS, -127, 0x80000000'00000000'00000000'00000000_u128}, // 1.0
      {Sign::NEG, -128, 0x80000000'00000000'00000000'00000000_u128}, // 1/2
      {Sign::POS, -132, 0xaaaaaaaa'aaaaaaaa'aaaaaaaa'aaaaaaab_u128}, // 1/4!
      {Sign::NEG, -137, 0xb60b60b6'0b60b60b'60b60b60'b60b60b6_u128}, // 1/6!
      {Sign::POS, -143, 0xd00d00d0'0d00d00d'00d00d00'd00d00d0_u128}, // 1/8!
      {Sign::NEG, -149, 0x93f27dbb'c4fae397'780b69f5'333c725b_u128}, // 1/10!
      {Sign::POS, -156, 0x8f76c77f'c6c4bdaa'26d4c3d6'7f425f60_u128}, // 1/12!
  };

  Float128 sin_u = fputil::quick_mul(
      u_f128, fputil::polyeval(u_sq, SIN_COEFFS[0], SIN_COEFFS[1],
                               SIN_COEFFS[2], SIN_COEFFS[3], SIN_COEFFS[4],
                               SIN_COEFFS[5], SIN_COEFFS[6]));
  Float128 cos_u = fputil::polyeval(u_sq, COS_COEFFS[0], COS_COEFFS[1],
                                    COS_COEFFS[2], COS_COEFFS[3], COS_COEFFS[4],
                                    COS_COEFFS[5], COS_COEFFS[6]);

  auto get_sin_k = [](unsigned kk) -> Float128 {
    unsigned idx = (kk & 64) ? 64 - (kk & 63) : (kk & 63);
    Float128 ans = SIN_K_PI_OVER_128_F128[idx];
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

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared utilities and constants for integer-only double-precision exponential
/// functions.
///
//===----------------------------------------------------------------------===//
//
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/frac128.h"
#include "src/__support/frac64.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/macros/properties/types.h"

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_UTILS_H

namespace LIBC_NAMESPACE_DECL {
namespace math {
namespace static_rounding {

// All constants below are generated via Sollya

// > display = hexadecimal;
// > print(2+round(1/log(2), 128, RN));
// LSB(LOG2_E_F128) = 2^-127
LIBC_INLINE_VAR constexpr Frac128 LOG2_E_F128 =
    Frac128({0xbe87'fed0'691d'3e89, 0xb8aa'3b29'5c17'f0bb});

// 2^x for x from 0 to (0b1111/2^4) = 15/16
// > for i from 0 to 15 do {
//   print(1+round(2^(i/16), 128, RN));
// };
// LSB(EXP_MID4[i]) = 2^-127
LIBC_INLINE_VAR constexpr Frac128 EXP_MID[] = {
    Frac128({0x0000'0000'0000'0000, 0x8000'0000'0000'0000}),
    Frac128({0xc5c9'5b8c'2154'c1b2, 0x85aa'c367'cc48'7b14}),
    Frac128({0xfbe4'6287'58a5'3c90, 0x8b95'c1e3'ea8b'd6e6}),
    Frac128({0x0fd6'd8e0'ae5a'c9d8, 0x91c3'd373'ab11'c336}),
    Frac128({0x46ad'2318'2e42'f6f6, 0x9837'f051'8db8'a96f}),
    Frac128({0xa091'1f09'ebb9'fdd1, 0x9ef5'3260'91a1'11ad}),
    Frac128({0x1cbd'7f62'1710'701b, 0xa5fe'd6a9'b151'38ea}),
    Frac128({0x4980'a8c8'f59a'2ec4, 0xad58'3eea'42a1'4ac6}),
    Frac128({0x597d'89b3'754a'be9f, 0xb504'f333'f9de'6484}),
    Frac128({0xa881'1fb6'6d0f'af7a, 0xbd08'a39f'580c'36be}),
    Frac128({0x3e2a'd0c9'64dd'9f37, 0xc567'2a11'5506'dadd}),
    Frac128({0xe235'838f'95f2'c6ed, 0xce24'8c15'1f84'80e3}),
    Frac128({0x39a6'8bb9'902d'3fde, 0xd744'fcca'd69d'6af4}),
    Frac128({0x0658'9504'8dd3'33ca, 0xe0cc'deec'2a94'e111}),
    Frac128({0xd02d'75b3'706e'54fb, 0xeac0'c6e7'dd24'392e}),
    Frac128({0x7b9d'0c7a'ed98'0fc3, 0xf525'7d15'2486'cc2c}),
};

// 128-bit polynomial approximation of 2^x coefficients generated with Sollya:
// > P = fpminimax(2^x, 12, [|1, 128...|], [0, 1/16], absolute, fixed);
// Store the fractional part of the coefficients below
// > dirtyinfnorm(2^x - P(x), [0, 1/16]);
// 0x1.b328...p-117
// LSB(EXPF_COEFFS[i]) = 2^-128
LIBC_INLINE_VAR constexpr Frac128 EXP_COEFFS[] = {
    // degree-0 = 1, add back afterwards to reduce calc ops
    Frac128({0xc9e3'b398'033f'0902, 0xb172'17f7'd1cf'79ab}),
    Frac128({0xde2d'60e0'866c'6365, 0x3d7f'7bff'058b'1d50}),
    Frac128({0x99d3'ad04'd47d'0efe, 0x0e35'846b'8250'5fc5}),
    Frac128({0x399a'b423'1644'1b55, 0x0276'556d'f749'cee5}),
    Frac128({0x405f'ebef'0f76'011a, 0x0057'61ff'9e29'9cc4}),
    Frac128({0x1ac4'9999'c0ef'b9ab, 0x000a'1848'97c3'63c4}),
    Frac128({0xe6f8'52f9'914f'2d9a, 0x0000'ffe5'fe2c'4573}),
    Frac128({0x6056'28a6'5061'b43f, 0x0000'162c'0223'a816}),
    Frac128({0x6b99'3efe'2193'e54c, 0x0000'01b5'253d'0671}),
    Frac128({0x9c76'f49c'ed65'743a, 0x0000'001e'4cf8'0b70}),
    Frac128({0x1ac1'3330'6c4e'3d33, 0x0000'0001'e8ae'6938}),
    Frac128({0x219f'9904'1f29'5f13, 0x0000'0000'1cd9'af73}),
};

// 64-bit polynomial approximation of 2^x coefficients generated with Sollya:
// > P = fpminimax(2^x, 7, [|1, 64...|], [0, 1/16], absolute, fixed);
// Store the fractional part of the coefficients below
// > dirtyinfnorm(2^x - P(x), [0, 1/16]);
// 0x1.999e...p-67
// This is different from EXPF_COEFFS: EXPF_COEFFS is for approximating for x in
// range of [0, 1]
// LSB(EXP_COEFFS[i]) = 2^-64
LIBC_INLINE_VAR constexpr Frac64 EXP_64_COEFFS[] = {
    // degree-0 = 1, add back afterwards to reduce calc ops
    Frac64(0xb172'17f7'd1cf'7aac), Frac64(0x3d7f'7bff'0589'89ce),
    Frac64(0x0e35'846b'8328'7490), Frac64(0x0276'556d'c04e'4f50),
    Frac64(0x0057'6207'1b69'46c2), Frac64(0x000a'17b9'1240'dbe4),
    Frac64(0x0001'0575'65fa'e20c),
};

// Round the fractional result and combine it with its exponent.
template <typename TFrac>
LIBC_INLINE uint64_t exp_handle_rounding(TFrac result_frac, bool is_neg, int d,
                                         uint64_t e_y,
                                         [[maybe_unused]] int rounding) {
  constexpr bool IS_FRAC64_PATH = cpp::is_same<TFrac, Frac64>::value;

  uint64_t result_bits;
  bool is_sticky = false;
  if constexpr (IS_FRAC64_PATH) {
    result_bits = result_frac.val[0];
  } else {
    result_bits = result_frac.val[1];
    is_sticky = result_frac.val[0] != 0;
  }

  bool is_subnormal = is_neg && d >= 0;
  unsigned shift = is_subnormal ? static_cast<unsigned>(12 + d) : 11;
  uint64_t result = shift == 64 ? 0 : result_bits >> shift;
  uint64_t half = uint64_t(1) << (shift - 1);
  bool round_bit = (result_bits & half) != 0;
  is_sticky |= (result_bits & (half - 1)) != 0;

#ifdef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  rounding = FE_TONEAREST;
#endif
  if (rounding == FE_TONEAREST)
    result += round_bit && (is_sticky || (result & 1));
  else if (rounding == FE_UPWARD)
    result += round_bit || is_sticky;

  // The normal significand already includes the implicit leading bit.
  return is_subnormal ? result : result + ((e_y - (uint64_t(1) << 20)) << 32);
}

LIBC_ALWAYS_INLINE void exp_scale(uint64_t x_s_shifted, int x_e_unbiased,
                                  bool is_neg, Frac128 &l2y_r, uint64_t &e_y,
                                  int &d) {
  using FPBits = fputil::FPBits<double>;

  Frac128 x_s_frac({0, x_s_shifted});
  Frac128 x_ln2 = x_s_frac * LOG2_E_F128;

  uint64_t k = 0;
  if (x_e_unbiased >= -1) {
    int shift = 62 - x_e_unbiased;
    k = (x_ln2 >> (64 + shift)).val[0];
    l2y_r = x_ln2 << (64 - shift);
  } else {
    int shift = -x_e_unbiased - 2;
    l2y_r = (shift < 128) ? (x_ln2 >> shift) : Frac128{};
  }

  if (LIBC_UNLIKELY(is_neg)) {
    if (l2y_r.val[0] != 0 || l2y_r.val[1] != 0) {
      ++k;
      l2y_r = ~l2y_r + Frac128(1);
    }
  }

  if (is_neg)
    e_y = (FPBits::EXP_BIAS << 20) - static_cast<uint32_t>(k << 20);
  else
    e_y = (FPBits::EXP_BIAS << 20) + static_cast<uint32_t>(k << 20);

  d = static_cast<int>(k) - FPBits::EXP_BIAS;
}

LIBC_ALWAYS_INLINE Frac64 exp_fast_path(uint64_t l2y_r_hi) {
  // Extract the top 4 fractional bits for the LUT index.
  uint16_t x_mid = static_cast<uint16_t>((l2y_r_hi >> 60) & 0xf);

  // The remaining 60 bits are the polynomial input.
  uint64_t x_lo = l2y_r_hi & ((1ull << 60) - 1);

  // Fast path: 64-bit calculations first

  // Don't << 4, as the polynomial approximation is correct in range [0, 1/16],
  // and x_lo is already in that range.
  // LSB(x_lo_frac) = 2^-64
  Frac64 x_lo_frac(x_lo);

  // Optimizations brought over from exp2f_integer_utils.h (2-way Estrin-Horner)
  // Will fallback to normal fputil::polyeval conditions aren't met.

#if defined(LIBC_TYPES_HAS_INT128) && !defined(LIBC_MATH_HAS_SMALL_TABLES)
  Frac64 x2 = x_lo_frac * x_lo_frac;
  Frac64 p_even = fputil::polyeval(x2, EXP_64_COEFFS[0], EXP_64_COEFFS[2],
                                   EXP_64_COEFFS[4], EXP_64_COEFFS[6]);
  Frac64 p_odd = fputil::polyeval(x2, EXP_64_COEFFS[1], EXP_64_COEFFS[3],
                                  EXP_64_COEFFS[5]);
  Frac64 p = fputil::multiply_add(p_odd, x_lo_frac, p_even) * x_lo_frac;
#else  // !defined(LIBC_TYPES_HAS_INT128) || defined(LIBC_MATH_HAS_SMALL_TABLES)
  Frac64 p = x_lo_frac * fputil::polyeval(x_lo_frac, EXP_64_COEFFS[0],
                                          EXP_64_COEFFS[1], EXP_64_COEFFS[2],
                                          EXP_64_COEFFS[3], EXP_64_COEFFS[4],
                                          EXP_64_COEFFS[5], EXP_64_COEFFS[6]);
#endif // defined(LIBC_TYPES_HAS_INT128) && !defined(LIBC_MATH_HAS_SMALL_TABLES)

  // With:
  //  - p = 2^lo - 1 --> 2^lo = p + 1
  //  - mid_val = 2^mid = EXP_MID[x_mid]
  // We have:
  //  2^mid * 2^lo = mid_val * (p + 1)
  // The same applies for both of the 64-bit and 128-bit paths
  // (Workaround because we're dealing with fractional representation of things)
  Frac64 mid_val = EXP_MID[x_mid].to_frac64();
  Frac64 result = fputil::multiply_add(mid_val, p, mid_val);

  return result;
}

LIBC_ALWAYS_INLINE uint64_t exp_accurate_path(Frac128 l2y_r, bool is_neg, int d,
                                              uint64_t e_y, int rounding) {
  uint16_t x_mid = static_cast<uint16_t>((l2y_r.val[1] >> 60) & 0xf);

  Frac128 x_lo_frac = l2y_r;
  x_lo_frac.val[1] &= 0x0fff'ffff'ffff'ffff;

#if defined(LIBC_TYPES_HAS_INT128) && !defined(LIBC_MATH_HAS_SMALL_TABLES)
  Frac128 x2 = x_lo_frac * x_lo_frac;
  Frac128 p_even =
      fputil::polyeval(x2, EXP_COEFFS[0], EXP_COEFFS[2], EXP_COEFFS[4],
                       EXP_COEFFS[6], EXP_COEFFS[8], EXP_COEFFS[10]);
  Frac128 p_odd =
      fputil::polyeval(x2, EXP_COEFFS[1], EXP_COEFFS[3], EXP_COEFFS[5],
                       EXP_COEFFS[7], EXP_COEFFS[9], EXP_COEFFS[11]);
  Frac128 p = fputil::multiply_add(p_odd, x_lo_frac, p_even) * x_lo_frac;
#else  // !defined(LIBC_TYPES_HAS_INT128) || defined(LIBC_MATH_HAS_SMALL_TABLES)
  Frac128 p =
      x_lo_frac * fputil::polyeval(x_lo_frac, EXP_COEFFS[0], EXP_COEFFS[1],
                                   EXP_COEFFS[2], EXP_COEFFS[3], EXP_COEFFS[4],
                                   EXP_COEFFS[5], EXP_COEFFS[6], EXP_COEFFS[7],
                                   EXP_COEFFS[8], EXP_COEFFS[9], EXP_COEFFS[10],
                                   EXP_COEFFS[11]);
#endif // defined(LIBC_TYPES_HAS_INT128) && !defined(LIBC_MATH_HAS_SMALL_TABLES)

  Frac128 mid_val = EXP_MID[x_mid];
  Frac128 result = fputil::multiply_add(mid_val, p, mid_val);

  return exp_handle_rounding(result, is_neg, d, e_y, rounding);
}

} // namespace static_rounding
} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_UTILS_H

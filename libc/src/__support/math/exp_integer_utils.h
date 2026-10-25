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

// > display = hexadecimal;
// > print(2+round(1/log(2), 64, RN));
// LSB(LOG2_E_F64) = 2^-63
LIBC_INLINE_VAR constexpr Frac64 LOG2_E_F64 = Frac64(0xb8aa'3b29'5c17'f0bc);

// 2^x for x from 0 to (0b1111/2^4) = 15/16
// > for i from 0 to 15 do {
//   print(1+round(2^(i/16), 128, RN));
// };
// LSB(EXP_MID4[i]) = 2^-127
LIBC_INLINE_VAR constexpr Frac128 EXP_MID[] = {
    Frac128({0x0000'0000'0000'0000ULL, 0x8000'0000'0000'0000ULL}),
    Frac128({0xc5c9'5b8c'2154'c1b2ULL, 0x85aa'c367'cc48'7b14ULL}),
    Frac128({0xfbe4'6287'58a5'3c90ULL, 0x8b95'c1e3'ea8b'd6e6ULL}),
    Frac128({0x0fd6'd8e0'ae5a'c9d8ULL, 0x91c3'd373'ab11'c336ULL}),
    Frac128({0x46ad'2318'2e42'f6f6ULL, 0x9837'f051'8db8'a96fULL}),
    Frac128({0xa091'1f09'ebb9'fdd1ULL, 0x9ef5'3260'91a1'11adULL}),
    Frac128({0x1cbd'7f62'1710'701bULL, 0xa5fe'd6a9'b151'38eaULL}),
    Frac128({0x4980'a8c8'f59a'2ec4ULL, 0xad58'3eea'42a1'4ac6ULL}),
    Frac128({0x597d'89b3'754a'be9fULL, 0xb504'f333'f9de'6484ULL}),
    Frac128({0xa881'1fb6'6d0f'af7aULL, 0xbd08'a39f'580c'36beULL}),
    Frac128({0x3e2a'd0c9'64dd'9f37ULL, 0xc567'2a11'5506'daddULL}),
    Frac128({0xe235'838f'95f2'c6edULL, 0xce24'8c15'1f84'80e3ULL}),
    Frac128({0x39a6'8bb9'902d'3fdeULL, 0xd744'fcca'd69d'6af4ULL}),
    Frac128({0x0658'9504'8dd3'33caULL, 0xe0cc'deec'2a94'e111ULL}),
    Frac128({0xd02d'75b3'706e'54fbULL, 0xeac0'c6e7'dd24'392eULL}),
    Frac128({0x7b9d'0c7a'ed98'0fc3ULL, 0xf525'7d15'2486'cc2cULL}),
};

// 128-bit polynomial approximation of 2^x coefficients generated with Sollya:
// > P = fpminimax(2^x, 12, [|1, 128...|], [0, 1/16], absolute, fixed);
// Store the fractional part of the coefficients below
// > dirtyinfnorm(2^x - P(x), [0, 1/16]);
// 0x1.b328...p-117
// LSB(EXPF_COEFFS[i]) = 2^-128
LIBC_INLINE_VAR constexpr Frac128 EXP_COEFFS[] = {
    // degree-0 = 1, add back afterwards to reduce calc ops
    Frac128({0xc9e3'b398'033f'0902ULL, 0xb172'17f7'd1cf'79abULL}),
    Frac128({0xde2d'60e0'866c'6365ULL, 0x3d7f'7bff'058b'1d50ULL}),
    Frac128({0x99d3'ad04'd47d'0efeULL, 0x0e35'846b'8250'5fc5ULL}),
    Frac128({0x399a'b423'1644'1b55ULL, 0x0276'556d'f749'cee5ULL}),
    Frac128({0x405f'ebef'0f76'011aULL, 0x0057'61ff'9e29'9cc4ULL}),
    Frac128({0x1ac4'9999'c0ef'b9abULL, 0x000a'1848'97c3'63c4ULL}),
    Frac128({0xe6f8'52f9'914f'2d9aULL, 0x0000'ffe5'fe2c'4573ULL}),
    Frac128({0x6056'28a6'5061'b43fULL, 0x0000'162c'0223'a816ULL}),
    Frac128({0x6b99'3efe'2193'e54cULL, 0x0000'01b5'253d'0671ULL}),
    Frac128({0x9c76'f49c'ed65'743aULL, 0x0000'001e'4cf8'0b70ULL}),
    Frac128({0x1ac1'3330'6c4e'3d33ULL, 0x0000'0001'e8ae'6938ULL}),
    Frac128({0x219f'9904'1f29'5f13ULL, 0x0000'0000'1cd9'af73ULL}),
};

// 64-bit polynomial approximation of 2^x coefficients generated with Sollya:
// > P = fpminimax(2^x, 6, [|1, 64...|], [0, 1/16], absolute, fixed);
// Store the fractional part of the coefficients below
// > dirtyinfnorm(2^x - P(x), [0, 1/16]);
// 0x1.2ac9...p-57
// This is different from EXPF_COEFFS: EXPF_COEFFS is for approximating for x in
// range of [0, 1]
// LSB(EXP_COEFFS[i]) = 2^-64
LIBC_INLINE_VAR constexpr Frac64 EXP_64_COEFFS[] = {
    // degree-0 = 1, add back afterwards to reduce calc ops
    Frac64(0xb172'17f7'd1cd'3e3cULL), Frac64(0x3d7f'7bff'0838'4d4aULL),
    Frac64(0x0e35'846a'6f46'd462ULL), Frac64(0x0276'55a0'd536'1dd9ULL),
    Frac64(0x0057'5d3d'4b45'0056ULL), Frac64(0x000a'504b'13fe'e008ULL),
};

// Round the fractional result and combine it with its exponent.
template <typename TFrac, typename TUInt>
LIBC_INLINE uint64_t exp_handle_rounding(TFrac result_frac, bool is_neg, int d,
                                         TUInt e_y,
                                         [[maybe_unused]] int rounding) {
  static_assert(cpp::is_same<TFrac, Frac64>::value ||
                cpp::is_same<TFrac, Frac128>::value);
  constexpr bool IS_FAST_PATH = cpp::is_same<TFrac, Frac64>::value;

  uint32_t shift_length = 11;
  uint64_t leading_one = 0;

  // subnormal
  if (LIBC_UNLIKELY(is_neg && d >= 0)) {
    e_y = 0;
    leading_one = uint64_t(1) << (52 - d);

    // Truncate the last 2 bits to avoid undefined behavior when shifting by 64
    // bits
    if (d >= 51) {
      d -= 2;
      if constexpr (IS_FAST_PATH)
        result_frac.val[0] >>= 2;
      else
        result_frac.val[1] >>= 2;
    }

    shift_length += d + 1;
  }

  // Get the bits, discarding the leading bit
  auto frac_bits = [&]() -> uint64_t {
    if constexpr (IS_FAST_PATH)
      return result_frac.val[0] << 1;
    else
      return (result_frac.val[1] << 1) | (result_frac.val[0] >> 63);
  };

#ifdef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  TUInt result = static_cast<TUInt>(frac_bits() >> shift_length) +
                 static_cast<TUInt>(leading_one);
  bool round_bit = (result & 1) != 0;
  result >>= 1;
  bool sticky = (frac_bits() & ((uint64_t(1) << shift_length) - 1)) != 0;
  if constexpr (!IS_FAST_PATH)
    sticky = sticky || (result_frac.val[0] != 0);
  if (round_bit && (sticky || (result & 1)))
    ++result;
  result += static_cast<TUInt>(e_y) << 32;

  return result;
#else  // LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  if (rounding == FE_TONEAREST) {
    TUInt result = static_cast<TUInt>(frac_bits() >> shift_length) +
                   static_cast<TUInt>(leading_one);
    bool round_bit = (result & 1) != 0;
    result >>= 1;
    bool sticky = (frac_bits() & ((uint64_t(1) << shift_length) - 1)) != 0;
    if constexpr (!IS_FAST_PATH)
      sticky = sticky || (result_frac.val[0] != 0);
    if (round_bit && (sticky || (result & 1)))
      ++result;
    result += static_cast<TUInt>(e_y) << 32;

    return result;
  }

  TUInt should_round_up = 0;

  if (LIBC_UNLIKELY(rounding == FE_UPWARD)) {
    uint64_t round_up_mask = (uint64_t(1) << (shift_length + 1)) - 1;
    bool has_remainder = (frac_bits() & round_up_mask) != 0;
    if constexpr (!IS_FAST_PATH) {
      has_remainder = has_remainder || (result_frac.val[0] != 0);
    }
    should_round_up = static_cast<TUInt>(has_remainder);
  }

  TUInt result = (static_cast<TUInt>(frac_bits() >> (shift_length + 1)) +
                  should_round_up + (leading_one >> 1));
  result += static_cast<TUInt>(e_y) << 32;

  return result;
#endif // LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
}

LIBC_ALWAYS_INLINE uint64_t exp_accurate_path(uint64_t x_s_shifted,
                                              int x_e_unbiased, bool is_neg,
                                              int rounding) {
  using FPBits = fputil::FPBits<double>;

  // Recalculate everything in 128-bit precision, with the same idea as the
  // 64-bit path.
  //
  // TODO: any way not to recompute?

  Frac128 x_s_frac({0, x_s_shifted});
  Frac128 x_ln2 = x_s_frac * LOG2_E_F128;

  uint64_t k = 0;
  Frac128 l2y_r;
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

  uint64_t e_y;
  if (is_neg)
    e_y = (FPBits::EXP_BIAS << 20) - static_cast<uint32_t>(k << 20);
  else
    e_y = (FPBits::EXP_BIAS << 20) + static_cast<uint32_t>(k << 20);

  int d = static_cast<int>(k) - FPBits::EXP_BIAS;

  if (LIBC_UNLIKELY(is_neg && d >= 53))
    return 0.0;

  uint16_t x_mid = static_cast<uint16_t>((l2y_r.val[1] >> 60) & 0xf);

  Frac128 x_lo_frac = l2y_r;
  x_lo_frac.val[1] &= 0x0fff'ffff'ffff'ffffULL;

  Frac128 p =
      x_lo_frac * fputil::polyeval(x_lo_frac, EXP_COEFFS[0], EXP_COEFFS[1],
                                   EXP_COEFFS[2], EXP_COEFFS[3], EXP_COEFFS[4],
                                   EXP_COEFFS[5], EXP_COEFFS[6], EXP_COEFFS[7],
                                   EXP_COEFFS[8], EXP_COEFFS[9], EXP_COEFFS[10],
                                   EXP_COEFFS[11]);

  Frac128 mid_val = EXP_MID[x_mid];
  Frac128 result = fputil::multiply_add(mid_val, p, mid_val);

  return exp_handle_rounding(result, is_neg, d, e_y, rounding);
}

} // namespace static_rounding

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_UTILS_H

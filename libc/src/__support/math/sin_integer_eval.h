//===-- Implementation header for sin using integer-only --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_SIN_INTEGER_EVAL_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_SIN_INTEGER_EVAL_H

#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/big_int.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math_extras.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

namespace integer_only {

struct Frac128 : public UInt<128> {
  using UInt<128>::UInt;

  constexpr Frac128 operator~() const {
    Frac128 r;
    r.val[0] = ~val[0];
    r.val[1] = ~val[1];
    return r;
  }
  constexpr Frac128 operator+(const Frac128 &other) const {
    UInt<128> r = UInt<128>(*this) + (UInt<128>(other));

    return Frac128(r.val);
  }
  constexpr Frac128 operator-(const Frac128 &other) const {
    UInt<128> r = UInt<128>(*this) - (UInt<128>(other));

    return Frac128(r.val);
  }
  constexpr Frac128 operator*(const Frac128 &other) const {
    UInt<128> r = UInt<128>::quick_mul_hi(UInt<128>(other));

    return Frac128(r.val);
  }
};

LIBC_INLINE constexpr double sin(double x) {
  using FPBits = typename fputil::FPBits<double>;
  FPBits xbits(x);

  uint16_t x_e = xbits.get_biased_exponent();
  uint64_t x_u = xbits.get_mantissa();
  x_u |= uint64_t(1) << FPBits::FRACTION_LEN;

  Frac128 x_frac({0, 0});
  unsigned k = 0;
  bool is_neg = xbits.is_neg();
  bool x_frac_is_neg = false;

  // x < 0.5
  if (x_e < FPBits::EXP_BIAS - 1) {
    // |x| < 2^-26, sin(x) ~ x.
    if (LIBC_UNLIKELY(x_e < FPBits::EXP_BIAS - 26))
      return x;
    // Normalize so that the MSB is 0.5.
    x_u <<= 10;
    unsigned shifts = FPBits::EXP_BIAS - 2 - x_e;
    if (shifts > 0) {
      if (shifts > 10)
        x_frac.val[0] = (x_u << (64 - shifts));
      x_frac.val[1] = x_u >> shifts;
    } else {
      x_frac.val[1] = x_u;
    }
  } else {
    // x is inf or nan.
    if (LIBC_UNLIKELY(x_e > 2 * FPBits::EXP_BIAS)) {
      // Silence signaling NaNs
      if (xbits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      // sin(+-Inf) = NaN
      if (xbits.get_mantissa() == 0) {
        fputil::set_errno_if_required(EDOM);
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      // x is quiet NaN
      return x;
    }

    // 1280 + 64 bits of 2/pi, printed using MPFR.
    // Notice that if we store from the highest bytes to lowest bytes, it is
    // essentially having 2/pi in big-endian.  On the other hand, uint64_t type
    // that will be used for computations later are in little-endian.  So a few
    // bit-reversal instructions will be introduced when we extract the parts
    // out.  It's possble to skip the bit-reversal instructions entirely by
    // having this table presented in little-endian, meaning from the lowest
    // bytes to highest bytes.  The tradeoff will be a bit more complicated and
    // awkward in the computations of the index, but it might be worth it?
    // We also add 8 more bytes to extend to all non-negative exponents.
    constexpr unsigned TWO_OVER_PI_LENGTH = 1280 / 8 + 7;
    constexpr uint8_t TWO_OVER_PI[TWO_OVER_PI_LENGTH] = {
        0x1D, 0x36, 0xF4, 0x9A, 0x20, 0xBC, 0xCF, 0xF0, 0xAB, 0x6B, 0x7B, 0xFC,
        0x46, 0x30, 0x03, 0x56, 0x08, 0x5D, 0x8D, 0x1F, 0xB1, 0x5F, 0xFB, 0x6B,
        0xEA, 0x92, 0x52, 0x8A, 0xF7, 0x39, 0x07, 0x3D, 0x7B, 0xF1, 0xE5, 0xEB,
        0xC7, 0xBA, 0x27, 0x75, 0x2D, 0xEA, 0x5F, 0x9E, 0x66, 0x3F, 0x46, 0x4F,
        0xB7, 0x09, 0xCB, 0x27, 0xCF, 0x7E, 0x36, 0x6D, 0x1F, 0x6D, 0x0A, 0x5A,
        0x8B, 0x11, 0x2F, 0xEF, 0x0F, 0x98, 0x05, 0xDE, 0xFF, 0x97, 0xF8, 0x1F,
        0x3B, 0x28, 0xF9, 0xBD, 0x8B, 0x5F, 0x84, 0x9C, 0xF4, 0x39, 0x53, 0x83,
        0x39, 0xD6, 0x91, 0x39, 0x41, 0x7E, 0x5F, 0xB4, 0x26, 0x70, 0x9C, 0xE9,
        0x84, 0x44, 0xBB, 0x2E, 0xF5, 0x35, 0x82, 0xE8, 0x3E, 0xA7, 0x29, 0xB1,
        0x1C, 0xEB, 0x1D, 0xFE, 0x1C, 0x92, 0xD1, 0x09, 0xEA, 0x2E, 0x49, 0x06,
        0xE0, 0xD2, 0x4D, 0x42, 0x3A, 0x6E, 0x24, 0xB7, 0x61, 0xC5, 0xBB, 0xDE,
        0xAB, 0x63, 0x51, 0xFE, 0x41, 0x90, 0x43, 0x3C, 0x99, 0x95, 0x62, 0xDB,
        0xC0, 0xDD, 0x34, 0xF5, 0xD1, 0x57, 0x27, 0xFC, 0x29, 0x15, 0x44, 0x4E,
        0x6E, 0x83, 0xF9, 0xA2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };

    // Perform range reduction mod pi/2 - We do multiplication x * (2/pi)
    //
    // Let T[i] be the i'th byte of 2/pi expansion:
    // Then 2/pi = T[0] * 2^-8 + T[1] * 2^-16 + ...
    //           = sum_i T[i] * 2^(-8(i + 1))
    // To be able to drop all T[j] * 2^(-8(j + 1)) for small j < i, we will want
    //   ulp(x) * lsb(T[i - 1] * 2^(-8 * i)) >= 4 = 2^2  (since 4 * pi/2 = 2*pi)
    // So:
    //   2^(e - 52) * 2^(-8 * i) >= 2^2
    // Or equivalently,
    //   e - 54 - 8*i >= 0.
    // Define:
    //   i = floor( (e - 54)/8 ),
    // and let
    //   s = e - 54 - 8i >= 0.
    // Since we store the mantissa of x, which is 53 bits long in a 64 bit
    // integer, we have some wiggle room to shuffle the lsb of x.
    // By shifting mantissa of x_u to the left by s, the lsb of x_u will be:
    //   2^(e - 52 - s), for which, the product of lsb's is now exactly 4
    //   lsb(x_u) * 2^(-8 * i)) = 4.
    // This will allow us to compute the full product:
    //   x_u * (T[i] * 2^(-8(i + 1)) + ... ) in exact fixed point.
    // From the formula of i, in order for i >= 0, e >= 54.  To support all the
    // exponents e >= 0, we could add ceil(54 / 8) = 7 0x00 bytes and shift the
    // index by 7.
    unsigned e_num = x_e - FPBits::EXP_BIAS + 2; // e - 54 + 7*8
    // With
    //   i = floor( (e - 54) / 8 ),
    // the shifted-by-7 index is:
    //   j = i + 7 = floor( (e - 54) / 8 ) + 7
    // Since the 64-bit integer chunk will be form by T[j] ... T[j + 7],
    // and we store the table in the little-endian form, we will index to the
    // lowest part of the 64-bit integer chunk, which is:
    //   idx = the index of the T[j + 7] part.
    unsigned j = e_num >> 3;
    unsigned idx = (TWO_OVER_PI_LENGTH - 1 - j - 7);
    unsigned shift = e_num & 7; // s = e - 54 - 8*i
    x_u <<= shift;              // lsb(x_u) = 2^(e - 52 - s)
    UInt<64> x_u64(x_u);
    // Gather parts
    auto get_uint64 = [](const uint8_t *ptr) -> uint64_t {
      return ptr[0] | (uint64_t(ptr[1]) << 8) | (uint64_t(ptr[2]) << 16) |
             (uint64_t(ptr[3]) << 24) | (uint64_t(ptr[4]) << 32) |
             (uint64_t(ptr[5]) << 40) | (uint64_t(ptr[6]) << 48) |
             (uint64_t(ptr[7]) << 56);
    };
    // lsb(c0) = 2^(-8i - 64)
    uint64_t c0 = get_uint64(&TWO_OVER_PI[idx]);
    // lsb(p0) = lsb(x_u) * lsb(c0)
    //         = 2^(e - 52 - s) * 2^(-8i - 64)
    //         = 2^(-62)
    // msb(p0) = 2^(-62 + 63) = 2^1.
    uint64_t p0 = x_u * c0;
    // lsb(c1) = lsb(c0) * 2^-64 = 2^(-8i - 128)
    UInt<64> c1(get_uint64(&TWO_OVER_PI[idx - 8]));
    // lsb(c2) = lsb(c1) * 2^-64 = 2^(-8i - 192)
    UInt<64> c2(get_uint64(&TWO_OVER_PI[idx - 16]));
    // lsb(p1) = lsb(x_u) * lsb(c1) = 2^(-62 - 64) = 2^-126
    UInt<128> p1 = x_u64.ful_mul(c1);
    // lsb(p2) = lsb(x_u) * lsb(c2) * 2^64 = 2^-126
    UInt<128> p2(x_u64.quick_mul_hi(c2));
    UInt<128> sum = p1 + p2;
    sum.val[1] += p0;
    // Get the highest 2 bits.
    k = static_cast<unsigned>(sum.val[1] >> 62);
    bool round_bit = sum.val[1] & 0x2000'0000'0000'0000;
    // Shift so that the leading bit is 0.5.
    x_frac.val[0] = (sum.val[0] << 2);
    x_frac.val[1] = (sum.val[1] << 2) | (sum.val[0] >> 62);

    // Round to nearest k.
    if (round_bit) {
      // Flip the sign.
      x_frac_is_neg = true;
      ++k;
      // Fast approximation of `1 - x_frac` with error = -lsb(x_frac) = -2^-128.
      // Since in 2-complement, -x = ~x + lsb(x).
      x_frac = ~x_frac;
    }

    // Perform multiplication x_frac * pi/2
    constexpr Frac128 PI_OVER_2_M1(
        {0x898c'c517'01b8'39a2, 0x921f'b544'42d1'8469});
    x_frac = fputil::multiply_add(x_frac, PI_OVER_2_M1, x_frac);
  }

  // 128-bit fixed-point minimax polynomial approximation of sin(x) generated by
  // Sollya with:
  // > P = fpminimax(sin(x), [|1, 3, 5, 7, 9, 11, 13|], [|1, 128...|],
  //                 [0, pi/4], fixed);
  // > dirtyinfnorm( (sin(x) - P(x))/sin(x), [0, pi/4]);
  // 0x1.17a4...p-58
  constexpr Frac128 SIN_COEFF[] = {
      // Positive coefficients
      Frac128({0x321f'bc0b'b8ca'f059, 0x0222'2222'221e'eac3}), // x^5
      Frac128({0x0556'929e'ad60'7cb2, 0x0000'2e3b'c6ab'd75e}), // x^9
      Frac128({0x4c97'758e'92ac'214c, 0x0000'0000'aec7'1a39}), // x^13
      // Negative coefficients
      Frac128({0x91b3'96a3'd5c5'fd6a, 0x2aaa'aaaa'aaaa'8ff2}), // x^3
      Frac128({0x36aa'355c'3311'996d, 0x000d'00d0'0cdf'8c9b}), // x^7
      Frac128({0xa260'c74f'239d'd891, 0x0000'006b'9795'15a2}), // x^11
  };
  // 128-bit fixed-point minimax polynomial approximation of cos(x) generated by
  // Sollya with:
  // > P = fpminimax(cos(x), [|0, 2, 4, 6, 8, 10, 12|], [|1, 128...|],
  //                 [0, pi/4], fixed);
  // > dirtyinfnorm( (cos(x) - P(x))/cos(x), [0, pi/4]);
  // 0x1.269f...p-54
  constexpr Frac128 COS_COEFF[] = {
      // Positive coefficients
      Frac128({0x860a'3e6c'cc50'e0d8, 0x0aaa'aaaa'aa77'5c33}), // x^4
      Frac128({0x84b2'76a3'c971'e7b8, 0x0001'a019'f80a'8ad5}), // x^8
      Frac128({0xed56'891e'f750'c7a9, 0x0000'0008'dc50'133d}), // x^12
      // Negative coefficients
      Frac128({0x56f6'2e74'b16e'5555, 0x7fff'ffff'fffe'4bfe}), // x^2
      Frac128({0xa87a'8f81'7440'7dd6, 0x005b'05b0'58fc'6fed}), // x^6
      Frac128({0x0082'310d'4e65'6b1f, 0x0000'049f'7cff'73d2}), // x^10
  };

  // cos when k = 1, 3
  bool is_cos = ((k & 1) == 1);
  // flip sign when k = 2, 3
  is_neg = is_neg != ((k & 2) == 2);

  const Frac128 *coeffs = is_cos ? COS_COEFF : SIN_COEFF;

  Frac128 xsq = x_frac * x_frac;
  Frac128 x4 = xsq * xsq;
  Frac128 poly_pos = fputil::polyeval(x4, coeffs[0], coeffs[1], coeffs[2]);
  Frac128 poly_neg = fputil::polyeval(x4, coeffs[3], coeffs[4], coeffs[5]);
  Frac128 r;
  if (!is_cos) {
    is_neg = (is_neg != x_frac_is_neg);
    // sin(x) = x + x^5 * poly_pos - x^3 * poly_neg
    Frac128 x3 = xsq * x_frac;
    Frac128 x5 = x4 * x_frac;
    poly_pos = fputil::multiply_add(x5, poly_pos, x_frac);
    poly_neg = x3 * poly_neg;
    r = poly_pos - poly_neg;
  } else {
    // cos(x) = 1 - (x^2 * poly_neg - x^4 * poly_pos)
    poly_pos = x4 * poly_pos;
    poly_neg = xsq * poly_neg;
    r = poly_neg - poly_pos;
    // Approximating 1 - r.
    r = ~r;
  }

  // Worst-case for range reduction > 2^-61, so the top 64-bits should be
  // non-zero for non-zero output.
  if (r.val[1] == 0)
    return 0.0;

  unsigned n = cpp::countl_zero(r.val[1]);
  uint64_t result = r.val[1];
  if (n > 0) {
    result <<= n;
    result |= (r.val[0] >> (64 - n));
  }
  unsigned rounding = ((static_cast<unsigned>(result) & 0x400) > 0);
  result >>= 11;
  result += (static_cast<uint64_t>(1021 - n) << 52) + rounding;
  result |= (static_cast<uint64_t>(is_neg) << 63);

  return cpp::bit_cast<double>(result);
}

} // namespace integer_only

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_SIN_INTEGER_EVAL_H

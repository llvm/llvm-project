//===-- x87_long_double.c - ARM64EC soft x87 `long double` helpers --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ARM64EC is ABI compatible with x86-64 Windows, where `long double` is the
// x87 80-bit extended format held in a 16-byte, 16-byte-aligned slot.  There
// is no x87 FPU on ARM64, so the compiler keeps such values in memory and
// copies them around with integer instructions only; whenever a computation
// has to be carried out, it converts the value to `double` with
// `__arm64ec_load_x87_long_double()`, computes in `double`, and converts the
// result back with `__arm64ec_store_x87_long_double()`.
//
// The 16-byte slot (little endian, as used by both x86-64 and ARM64) is
//
//   offset  0: 64-bit significand, including the explicit integer bit
//   offset  8: 16-bit sign/exponent word: bit 15 is the sign, bits 14..0 is
//              the biased exponent
//   offset 10: 6 bytes of padding that are not read or written by the x87
//
// Note that the x87 extended format does not use a hidden bit: The
// significand of a normal number has bit 63 set, and for a denormal or zero
// bit 63 is clear.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

/// The memory image of an x87 extended-precision value.  Deliberately declared
/// as integers so that neither the compiler nor this file can end up using
/// floating-point instructions on it.
typedef struct {
  uint64_t significand;
  uint16_t sign_exponent;
  uint16_t padding[3];
} arm64ec_x87_image;

typedef union {
  uint64_t u;
  double d;
} arm64ec_double_bits;

#define ARM64EC_X87_EXP_BIAS 16383
#define ARM64EC_X87_EXP_MAX 0x7fffu
#define ARM64EC_DBL_EXP_MAX 0x7ffu

/// The x87 real indefinite, which is what it produces for an invalid operand.
/// Its sign bit is always set.
static double arm64ec_indefinite_qnan(void) {
  arm64ec_double_bits v;
  v.u = 0xfff8000000000000ull;
  return v.d;
}

static double arm64ec_make_double(unsigned sign, uint32_t biased_exponent,
                                  uint64_t fraction) {
  arm64ec_double_bits v;
  v.u = ((uint64_t)sign << 63) | ((uint64_t)biased_exponent << 52) | fraction;
  return v.d;
}

/// Convert an x87 extended-precision memory image to `double`, rounding to
/// nearest with ties to even.  This matches what the x87 itself does for
/// 'fld m80; fstp m64' with the precision control set to 64 bits.
double __arm64ec_load_x87_long_double(const arm64ec_x87_image *image) {
  uint64_t m = image->significand;
  unsigned sign = (unsigned)(image->sign_exponent >> 15);
  unsigned e = (unsigned)(image->sign_exponent & 0x7fff);
  int exponent;
  uint64_t kept, dropped;

  if (e == 0x7fff) {
    if (m == 0x8000000000000000ull)
      // Infinity.
      return arm64ec_make_double(sign, ARM64EC_DBL_EXP_MAX, 0);
    // A NaN whose integer bit is set is a real NaN: keep the top 52 bits of
    // its payload and make sure the quiet bit is set.  Anything else (a
    // pseudo-NaN or pseudo-infinity with the integer bit clear) is an invalid
    // encoding, for which the hardware produces the indefinite QNaN.
    if ((m >> 63) == 0)
      return arm64ec_indefinite_qnan();
    return arm64ec_make_double(sign, ARM64EC_DBL_EXP_MAX,
                               ((m >> 11) & 0xfffffffffffffull) |
                                   0x8000000000000ull);
  }

  // Non-zero biased exponent with the integer bit clear is invalid.
  if (e != 0 && (m >> 63) == 0)
    return arm64ec_indefinite_qnan();

  // A pseudo-denormal (biased exponent zero with the integer bit set) is a
  // perfectly valid representation of m * 2^-16445 and behaves like the
  // normal number with biased exponent one.  A denormalized number is far
  // below the range of `double` and therefore underflows to zero.
  if (e == 0) {
    e = (unsigned)(m >> 63);
    if (e == 0) // zero or denormal, both of which become a signed zero
      return arm64ec_make_double(sign, 0, 0);
  }

  // value == m * 2^(exponent - 63), with bit 63 of m set
  exponent = (int)e - ARM64EC_X87_EXP_BIAS;

  if (exponent > 1023)
    return arm64ec_make_double(sign, ARM64EC_DBL_EXP_MAX, 0);

  if (exponent >= -1022) {
    // A normal `double`: round the 64-bit significand to 53 bits.
    kept = m >> 11;
    dropped = m & 0x7ffu;
    if (dropped > 0x400u || (dropped == 0x400u && (kept & 1)))
      if (++kept == 0x20000000000000ull) {
        kept >>= 1;
        if (++exponent > 1023)
          return arm64ec_make_double(sign, ARM64EC_DBL_EXP_MAX, 0);
      }
    return arm64ec_make_double(sign, (uint32_t)(exponent + 1023),
                               kept & 0xfffffffffffffull);
  }

  // A subnormal `double`: the result is
  // round(m * 2^(exponent + 1011)) units of 2^-1074.
  int shift = -(exponent + 1011);
  uint64_t k;

  if (shift > 64)
    k = 0;
  else if (shift == 64) {
    // m is in [2^63, 2^64), so the quotient is in [0.5, 1) and only a value
    // strictly greater than one half rounds up.
    k = m > 0x8000000000000000ull ? 1 : 0;
  } else {
    uint64_t quotient = m >> shift;
    uint64_t remainder = m & ((1ull << shift) - 1);
    uint64_t half = 1ull << (shift - 1);
    if (remainder > half || (remainder == half && (quotient & 1)))
      ++quotient;
    k = quotient;
  }
  // 'k' may reach 2^52, which is exactly the smallest normal `double`; it
  // then carries into the exponent field on its own.
  return arm64ec_make_double(sign, 0, k);
}

/// Convert a `double` to an x87 extended-precision memory image.  Every
/// `double` is exactly representable in the x87 format, so this never rounds.
/// Padding bits are undefined.
void __arm64ec_store_x87_long_double(arm64ec_x87_image *image, double value) {
  arm64ec_double_bits v;
  unsigned sign;
  unsigned e;
  uint64_t m;

  v.d = value;
  sign = (unsigned)(v.u >> 63);
  e = (unsigned)((v.u >> 52) & 0x7ff);
  m = v.u & 0xfffffffffffffull;

  if (e == ARM64EC_DBL_EXP_MAX) {
    if (m == 0)
      image->significand = 0x8000000000000000ull; // infinity
    else
      // Map the 52-bit `double` payload onto the top bits of the 63-bit x87
      // fraction.  Like 'fld m64', quiet a signalling NaN.
      image->significand = (m << 11) | 0xc000000000000000ull;
    image->sign_exponent = (uint16_t)((sign << 15) | ARM64EC_X87_EXP_MAX);
  } else if (e == 0) {
    if (m == 0) {
      image->significand = 0; // zero
      image->sign_exponent = (uint16_t)(sign << 15);
    } else {
      // A subnormal `double` has to be normalized: it is
      // 1.f * 2^(p - 1074) where p is the index of its top set bit.
      int p = 63 - __builtin_clzll(m);
      image->significand = m << (63 - p);
      image->sign_exponent = (uint16_t)((sign << 15) | (unsigned)(p + 15309));
    }
  } else {
    // Shift the 52-bit `double` fraction up to the top of the 63-bit x87
    // fraction and add the explicit integer bit.
    image->significand = (m << 11) | 0x8000000000000000ull;
    image->sign_exponent = (uint16_t)((sign << 15) | (e + 15360));
  }
}

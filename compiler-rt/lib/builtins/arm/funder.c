//===-- funder.c - Handle single-precision floating-point underflow -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This helper function is available for use by single-precision float
// arithmetic implementations to handle underflowed output values, if they were
// computed in the form of a normalized mantissa and an out-of-range exponent.
//
// On input: x should be a complete IEEE 754 floating-point value representing
// the desired output scaled up by 2^192 (the same value that would have been
// passed to an underflow trap handler in IEEE 754:1985).
//
// This isn't enough information to re-round to the correct output denormal
// without also knowing whether x itself has already been rounded, and which
// way. 'errsign' gives this information, by indicating the sign of the value
// (true result - x). That is, if errsign > 0 it means the true value was
// larger (x was rounded down); if errsign < 0 then x was rounded up; if
// errsign == 0 then x represents the _exact_ desired output value.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#define SIGNBIT 0x80000000
#define MANTSIZE 23
#define BIAS 0xc0

uint32_t __compiler_rt_funder(uint32_t x, uint32_t errsign) {
  uint32_t sign = x & SIGNBIT;
  uint32_t exponent = (x << 1) >> 24;

  // Rule out exponents so small (or large!) that no denormalisation
  // is needed.
  if (exponent > BIAS) {
    // Exponent 0xc1 or above means a normalised number got here by
    // mistake, so we just remove the 0xc0 exponent bias and go
    // straight home.
    return x - (BIAS << MANTSIZE);
  }
  uint32_t bits_lost = BIAS + 1 - exponent;
  if (bits_lost > MANTSIZE + 1) {
    // The implicit leading 1 of the intermediate value's mantissa is
    // below the lowest mantissa bit of a denormal by at least 2 bits.
    // Round down to 0 unconditionally.
    return sign;
  }

  // Make the full mantissa (with leading bit) at the top of the word.
  uint32_t mantissa = 0x80000000 | (x << 8);
  // Adjust by 1 depending on the sign of the error.
  mantissa -= errsign >> 31;
  mantissa += (-errsign) >> 31;

  // Shift down to the output position, keeping the bits shifted off.
  uint32_t outmant, shifted_off;
  if (bits_lost == MANTSIZE + 1) {
    // Special case for the exponent where we have to shift the whole
    // of 'mantissa' off the bottom of the word.
    outmant = 0;
    shifted_off = mantissa;
  } else {
    outmant = mantissa >> (8 + bits_lost);
    shifted_off = mantissa << (32 - (8 + bits_lost));
  }

  // Re-round.
  if (shifted_off >> 31) {
    outmant++;
    if (!(shifted_off << 1))
      outmant &= ~1; // halfway case: round to even
  }

  return sign | outmant;
}

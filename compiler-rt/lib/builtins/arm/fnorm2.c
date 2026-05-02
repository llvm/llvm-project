//===-- fnorm2.c - Handle single-precision denormal inputs to binary op ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This helper function is available for use by single-precision float
// arithmetic implementations, to handle denormal inputs on entry by
// renormalizing the mantissa and modifying the exponent to match.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

// Structure containing the function's inputs and outputs.
//
// On entry: a, b are two input floating-point numbers, still in IEEE 754
// encoding. expa and expb are the 8-bit exponents of those numbers, extracted
// and shifted down to the low 8 bits of the word, with no other change.
// Neither value should be zero, or have the maximum exponent (indicating an
// infinity or NaN).
//
// On exit: each of a and b contains the mantissa of the input value, with the
// leading 1 bit made explicit, and shifted up to the top of the word. If expa
// was zero (indicating that a was denormal) then it is now represented as a
// normalized number with an out-of-range exponent (zero or negative). The same
// applies to expb and b.
struct fnorm2 {
  uint32_t a, b, expa, expb;
};

void __compiler_rt_fnorm2(struct fnorm2 *values) {
  // Shift the mantissas of a and b to the right place to follow a leading 1 in
  // the top bit, if there is one.
  values->a <<= 8;
  values->b <<= 8;

  // Test if a is denormal.
  if (values->expa == 0) {
    // If so, decide how much further up to shift its mantissa, and adjust its
    // exponent to match. This brings the leading 1 of the denormal mantissa to
    // the top of values->a.
    uint32_t shift = __builtin_clz(values->a);
    values->a <<= shift;
    values->expa = 1 - shift;
  } else {
    // Otherwise, leave the mantissa of a in its current position, and OR in
    // the explicit leading 1.
    values->a |= 0x80000000;
  }

  // Do the same operation on b.
  if (values->expb == 0) {
    uint32_t shift = __builtin_clz(values->b);
    values->b <<= shift;
    values->expb = 1 - shift;
  } else {
    values->b |= 0x80000000;
  }
}

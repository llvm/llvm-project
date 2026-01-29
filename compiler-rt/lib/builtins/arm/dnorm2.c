//===-- dnorm2.c - Handle double-precision denormal inputs to binary op ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This helper function is available for use by double-precision float
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
// leading 1 bit made explicit, and shifted up to bit 52 (the same place it
// would have been if the number was normalized already). If expa was zero
// (indicating that a was denormal) then it is now represented as a normalized
// number with an out-of-range exponent (zero or negative). The same applies to
// expb and b.
//
// The sign bits from the input floating-point numbers are discarded
// completely. The caller is expected to have stored those somewhere
// safe already.
struct dnorm2 {
  uint64_t a, b;
  uint32_t expa, expb;
};

void __compiler_rt_dnorm2(struct dnorm2 *values) {
  values->a &= ~0xFFF0000000000000;
  values->b &= ~0xFFF0000000000000;

  if (values->expa == 0) {
    unsigned shift = __builtin_clzll(values->a) - 11;
    values->a <<= shift;
    values->expa = 1 - shift;
  } else {
    values->a |= 0x0010000000000000;
  }

  if (values->expb == 0) {
    unsigned shift = __builtin_clzll(values->b) - 11;
    values->b <<= shift;
    values->expb = 1 - shift;
  } else {
    values->b |= 0x0010000000000000;
  }
}

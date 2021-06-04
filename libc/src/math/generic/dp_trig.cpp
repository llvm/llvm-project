//===-- Utilities for double precision trigonometric functions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/FPUtil/UInt.h"
#include "src/__support/FPUtil/XFloat.h"

using FPBits = __llvm_libc::fputil::FPBits<double>;

namespace __llvm_libc {

// Implementation is based on the Payne and Hanek range reduction algorithm.
// The caller should ensure that x is positive.
// Consider:
//   x/y = x * 1/y = I + F
// I is the integral part and F the fractional part of the result of the
// division operation. Then M = mod(x, y) = F * y. In order to compute M, we
// first compute F. We do it by dropping bits from 1/y which would only
// contribute integral results in the operation x * 1/y. This helps us get
// accurate values of F even when x is a very large number.
//
// Internal operations are performed at 192 bits of precision.
static double mod_impl(double x, const uint64_t y_bits[3],
                       const uint64_t inv_y_bits[20], int y_exponent,
                       int inv_y_exponent) {
  FPBits bits(x);
  int exponent = bits.getExponent();
  int bit_drop = (exponent - 52) + inv_y_exponent + 1;
  bit_drop = bit_drop >= 0 ? bit_drop : 0;
  int word_drop = bit_drop / 64;
  bit_drop %= 64;
  fputil::UInt<256> man4;
  for (size_t i = 0; i < 4; ++i) {
    man4[3 - i] = inv_y_bits[word_drop + i];
  }
  man4.shift_left(bit_drop);
  fputil::UInt<192> man_bits;
  for (size_t i = 0; i < 3; ++i)
    man_bits[i] = man4[i + 1];
  fputil::XFloat<192> result(inv_y_exponent - word_drop * 64 - bit_drop,
                             man_bits);
  result.mul(x);
  result.drop_int(); // |result| now holds fractional part of x/y.

  fputil::UInt<192> y_man;
  for (size_t i = 0; i < 3; ++i)
    y_man[i] = y_bits[2 - i];
  fputil::XFloat<192> y_192(y_exponent, y_man);
  return result.mul(y_192);
}

static constexpr int TwoPIExponent = 2;

// The mantissa bits of 2 * PI.
// The most signification bits are in the first uint64_t word
// and the least signification bits are in the last word. The
// first word includes the implicit '1' bit.
static constexpr uint64_t TwoPI[] = {0xc90fdaa22168c234, 0xc4c6628b80dc1cd1,
                                     0x29024e088a67cc74};

static constexpr int InvTwoPIExponent = -3;

// The mantissa bits of 1/(2 * PI).
// The most signification bits are in the first uint64_t word
// and the least signification bits are in the last word. The
// first word includes the implicit '1' bit.
static constexpr uint64_t InvTwoPI[] = {
    0xa2f9836e4e441529, 0xfc2757d1f534ddc0, 0xdb6295993c439041,
    0xfe5163abdebbc561, 0xb7246e3a424dd2e0, 0x6492eea09d1921c,
    0xfe1deb1cb129a73e, 0xe88235f52ebb4484, 0xe99c7026b45f7e41,
    0x3991d639835339f4, 0x9c845f8bbdf9283b, 0x1ff897ffde05980f,
    0xef2f118b5a0a6d1f, 0x6d367ecf27cb09b7, 0x4f463f669e5fea2d,
    0x7527bac7ebe5f17b, 0x3d0739f78a5292ea, 0x6bfb5fb11f8d5d08,
    0x56033046fc7b6bab, 0xf0cfbc209af4361e};

double mod_2pi(double x) {
  static constexpr double _2pi = 6.283185307179586;
  if (x < _2pi)
    return x;
  return mod_impl(x, TwoPI, InvTwoPI, TwoPIExponent, InvTwoPIExponent);
}

// Returns mod(x, pi/2)
double mod_pi_over_2(double x) {
  static constexpr double pi_over_2 = 1.5707963267948966;
  if (x < pi_over_2)
    return x;
  return mod_impl(x, TwoPI, InvTwoPI, TwoPIExponent - 2, InvTwoPIExponent + 2);
}

// Returns mod(x, pi/4)
double mod_pi_over_4(double x) {
  static constexpr double pi_over_4 = 0.7853981633974483;
  if (x < pi_over_4)
    return x;
  return mod_impl(x, TwoPI, InvTwoPI, TwoPIExponent - 3, InvTwoPIExponent + 3);
}

} // namespace __llvm_libc

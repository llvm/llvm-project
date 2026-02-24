//===-- udivmodoi4.c - Implement __udivmodoi4 -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __udivmodoi4 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_256BIT

// Returns the 256 bit division result by 128 bit. Result must fit in 128 bits.
// Remainder stored in r.
// Adapted from the 128/64 algorithm in udivmodti4.c.
UNUSED
static inline tu_int udiv256by128to128default(tu_int u1, tu_int u0, tu_int v,
                                              tu_int *r) {
  const unsigned n_utword_bits = sizeof(tu_int) * CHAR_BIT;
  const tu_int b = (tu_int)1 << (n_utword_bits / 2); // Number base (64 bits)
  tu_int un1, un0;                                   // Norm. dividend LSD's
  tu_int vn1, vn0;                                   // Norm. divisor digits
  tu_int q1, q0;                                     // Quotient digits
  tu_int un128, un21, un10;                          // Dividend digit pairs
  tu_int rhat;                                       // A remainder
  si_int s; // Shift amount for normalization

  s = __clzti2(v);
  if (s > 0) {
    // Normalize the divisor.
    v = v << s;
    un128 = (u1 << s) | (u0 >> (n_utword_bits - s));
    un10 = u0 << s;
  } else {
    // Avoid undefined behavior of (u0 >> 128).
    un128 = u1;
    un10 = u0;
  }

  // Break divisor up into two 64-bit digits.
  vn1 = v >> (n_utword_bits / 2);
  vn0 = v & (((tu_int)1 << (n_utword_bits / 2)) - 1);

  // Break right half of dividend into two digits.
  un1 = un10 >> (n_utword_bits / 2);
  un0 = un10 & (((tu_int)1 << (n_utword_bits / 2)) - 1);

  // Compute the first quotient digit, q1.
  q1 = un128 / vn1;
  rhat = un128 - q1 * vn1;

  // q1 has at most error 2. No more than 2 iterations.
  while (q1 >= b || q1 * vn0 > b * rhat + un1) {
    q1 = q1 - 1;
    rhat = rhat + vn1;
    if (rhat >= b)
      break;
  }

  un21 = un128 * b + un1 - q1 * v;

  // Compute the second quotient digit.
  q0 = un21 / vn1;
  rhat = un21 - q0 * vn1;

  // q0 has at most error 2. No more than 2 iterations.
  while (q0 >= b || q0 * vn0 > b * rhat + un0) {
    q0 = q0 - 1;
    rhat = rhat + vn1;
    if (rhat >= b)
      break;
  }

  *r = (un21 * b + un0 - q0 * v) >> s;
  return q1 * b + q0;
}

static inline tu_int udiv256by128to128(tu_int u1, tu_int u0, tu_int v,
                                       tu_int *r) {
  return udiv256by128to128default(u1, u0, v, r);
}

// Effects: if rem != 0, *rem = a % b
// Returns: a / b

COMPILER_RT_ABI ou_int __udivmodoi4(ou_int a, ou_int b, ou_int *rem) {
  const unsigned n_uoword_bits = sizeof(ou_int) * CHAR_BIT;
  uowords dividend;
  dividend.all = a;
  uowords divisor;
  divisor.all = b;
  uowords quotient;
  uowords remainder;
  if (divisor.all > dividend.all) {
    if (rem)
      *rem = dividend.all;
    return 0;
  }
  // When the divisor fits in 128 bits, we can use an optimized path.
  if (divisor.s.high == 0) {
    remainder.s.high = 0;
    if (dividend.s.high < divisor.s.low) {
      // The result fits in 128 bits.
      quotient.s.low = udiv256by128to128(dividend.s.high, dividend.s.low,
                                         divisor.s.low, &remainder.s.low);
      quotient.s.high = 0;
    } else {
      // First, divide with the high part to get the remainder in
      // dividend.s.high. After that dividend.s.high < divisor.s.low.
      quotient.s.high = dividend.s.high / divisor.s.low;
      dividend.s.high = dividend.s.high % divisor.s.low;
      quotient.s.low = udiv256by128to128(dividend.s.high, dividend.s.low,
                                         divisor.s.low, &remainder.s.low);
    }
    if (rem)
      *rem = remainder.all;
    return quotient.all;
  }
  // 0 <= shift <= 127.
  si_int shift = __clzti2(divisor.s.high) - __clzti2(dividend.s.high);
  divisor.all <<= shift;
  quotient.s.high = 0;
  quotient.s.low = 0;
  for (; shift >= 0; --shift) {
    quotient.s.low <<= 1;
    // Branch free version of.
    // if (dividend.all >= divisor.all)
    // {
    //    dividend.all -= divisor.all;
    //    carry = 1;
    // }
    const oi_int s =
        (oi_int)(divisor.all - dividend.all - 1) >> (n_uoword_bits - 1);
    quotient.s.low |= s & 1;
    dividend.all -= divisor.all & s;
    divisor.all >>= 1;
  }
  if (rem)
    *rem = dividend.all;
  return quotient.all;
}

#endif // CRT_HAS_256BIT

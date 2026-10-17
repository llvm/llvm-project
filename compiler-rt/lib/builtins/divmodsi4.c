//===-- divmodsi4.c - Implement __divmodsi4
//--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __divmodsi4 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#if defined(__AVR__)
// On AVR, as with __udivmodsi4, the quotient is returned in the low half of a
// 64-bit value and the remainder in its high half.

// Returns: a / b in the low half, a % b in the high half

COMPILER_RT_ABI di_int __divmodsi4(si_int a, si_int b) {
  const int bits_in_word_m1 = (int)(sizeof(si_int) * CHAR_BIT) - 1;
  si_int s_a = a >> bits_in_word_m1; // s_a = a < 0 ? -1 : 0
  si_int s_b = b >> bits_in_word_m1; // s_b = b < 0 ? -1 : 0
  a = (su_int)(a ^ s_a) - s_a;       // negate if s_a == -1
  b = (su_int)(b ^ s_b) - s_b;       // negate if s_b == -1
  s_b ^= s_a;                        // sign of quotient
  dwords r;
  r.all = __udivmodsi4(a, b);
  r.s.low = (r.s.low ^ s_b) - s_b;   // negate if s_b == -1
  r.s.high = (r.s.high ^ s_a) - s_a; // negate if s_a == -1
  return r.all;
}
#else
// Returns: a / b, *rem = a % b

COMPILER_RT_ABI si_int __divmodsi4(si_int a, si_int b, si_int *rem) {
  const int bits_in_word_m1 = (int)(sizeof(si_int) * CHAR_BIT) - 1;
  si_int s_a = a >> bits_in_word_m1;                    // s_a = a < 0 ? -1 : 0
  si_int s_b = b >> bits_in_word_m1;                    // s_b = b < 0 ? -1 : 0
  a = (su_int)(a ^ s_a) - s_a;                          // negate if s_a == -1
  b = (su_int)(b ^ s_b) - s_b;                          // negate if s_b == -1
  s_b ^= s_a;                                           // sign of quotient
  su_int r;
  si_int q = (__udivmodsi4(a, b, &r) ^ s_b) - s_b;      // negate if s_b == -1
  *rem = (r ^ s_a) - s_a;                               // negate if s_a == -1
  return q;
}
#endif

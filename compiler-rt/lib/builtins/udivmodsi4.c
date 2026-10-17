//===-- udivmodsi4.c - Implement __udivmodsi4 -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __udivmodsi4 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#if defined(__AVR__)
// On AVR, GCC and Clang call __udivmodsi4 without a remainder pointer, and
// expect the quotient back in R18-R21 and the remainder in R22-R25, which is
// where a 64-bit value with the quotient in its low half is returned.

// Returns: a / b in the low half, a % b in the high half

COMPILER_RT_ABI du_int __udivmodsi4(su_int a, su_int b) {
  udwords r;
  r.s.low = __udivsi3(a, b);
  r.s.high = a - r.s.low * b;
  return r.all;
}
#else
// Returns: a / b, *rem = a % b

COMPILER_RT_ABI su_int __udivmodsi4(su_int a, su_int b, su_int *rem) {
  si_int d = __udivsi3(a, b);
  *rem = a - (d * b);
  return d;
}
#endif

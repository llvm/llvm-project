//===-- fnan2.c - Handle single-precision NaN inputs to binary operation --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This helper function is available for use by single-precision float
// arithmetic implementations to handle propagating NaNs from the input
// operands to the output, in a way that matches Arm hardware FP.
//
// On input, a and b are floating-point numbers in IEEE 754 encoding, and at
// least one of them must be a NaN. The return value is the correct output NaN.
//
//===----------------------------------------------------------------------===//

unsigned __fnan2(unsigned a, unsigned b) {
  unsigned aadj = (a << 1) + 0x00800000;
  unsigned badj = (b << 1) + 0x00800000;
  if (aadj > 0xff800000)
    return a | 0x00400000;
  if (badj > 0xff800000)
    return b | 0x00400000;
  if (aadj < 0x00800000)
    return a;
  else /* expect (badj < 0x00800000) */
    return b;
}

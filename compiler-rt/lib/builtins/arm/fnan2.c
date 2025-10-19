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
// A signalling NaN in the input (with bit 22 clear) takes priority over any
// quiet NaN, and is adjusted on return by setting bit 22 to make it quiet. If
// both inputs are the same type of NaN then the first input takes priority:
// the input a is used instead of b.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

uint32_t __compiler_rt_fnan2(uint32_t a, uint32_t b) {
  // Make shifted-left copies of a and b to discard the sign bit. Then add 1 at
  // the bit position where the quiet vs signalling bit ended up. This squashes
  // all the signalling NaNs to the top of the range of 32-bit values, from
  // 0xff800001 to 0xffffffff inclusive; meanwhile, all the quiet NaN values
  // wrap round to the bottom, from 0 to 0x007fffff inclusive. So we can detect
  // a signalling NaN by asking if it's greater than 0xff800000, and a quiet
  // one by asking if it's less than 0x00800000.
  uint32_t aadj = (a << 1) + 0x00800000;
  uint32_t badj = (b << 1) + 0x00800000;
  if (aadj > 0xff800000)   // a is a signalling NaN?
    return a | 0x00400000; //   if so, return it with the quiet bit set
  if (badj > 0xff800000)   // b is a signalling NaN?
    return b | 0x00400000; //   if so, return it with the quiet bit set
  if (aadj < 0x00800000)   // a is a quiet NaN?
    return a;              // if so, return it
  return b;                // otherwise we expect b must be a quiet NaN
}

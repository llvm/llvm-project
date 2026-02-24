//===-- floatsidf.S - 32-bit signed int to double-precision FP conversion -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __floatsidf function (32-bit signed integer to
// double precision floating point conversion), with the IEEE-754 default
// rounding (to nearest, ties to even), for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"
#include "endian.h"

  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__floatsidf)
  push {r4, lr}
  bl __aeabi_i2d
  VMOV_TO_DOUBLE(d0, r0, r1)
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__floatsidf, __aeabi_i2d)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_i2d)

  // Start by separating the input into its sign and its absolute value.
  ands    r2, r0, #0x80000000   // get sign
  rsbmi   r0, r0, #0            // negate input if it was negative

  // Shift the leading bit of the input to the top of xh. In the
  // process, we detect a zero input, and branch out of line for it.
  clz     r3, r0
  lsls    xh, r0, r3
  beq     LOCAL_LABEL(zero)

  // Use the shift count in r3 to calculate the output exponent, and combine it
  // with the sign bit in r2.
  //
  // If r3=0 then the input value was in the range [2^31,2^32), so its exponent
  // in double precision should be 0x41e. We want to reduce that by 1 so that
  // the leading bit of the mantissa will increment it when we add it in. So
  // the exponent should be 0x41d minus r3.
  //
  // (For a signed integer, the only possible value in that range would be
  // 2^31, obtained as the absolute value of an original input of -2^31.)
  rsb     r3, r3, #0x1d         // 0x1d minus shift count
  add     r3, r3, #0x400        // 0x41d minus shift count
  add     r2, r2, r3, lsl #20   // r2 is now sign+exponent

  // Shift the mantissa left and right to get the parts that go in xl and xh,
  // and combine with the sign and exponent we just computed.
  lsl     xl, xh, #21           // low bits of mantissa
  add     xh, r2, xh, lsr #11   // high bits of mantissa + sign + exponent
  bx      lr

LOCAL_LABEL(zero):
  // We come here if the input was zero. We've just set xh to 0, so we
  // only need to set xl to 0 too and return.
  mov     xl, #0
  bx      lr

END_COMPILERRT_FUNCTION(__aeabi_i2d)

NO_EXEC_STACK_DIRECTIVE

//===-- fixunssfdi.S - single-precision FP to 64-bit unsigned int conversion==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __fixunssfdi function (single precision floating
// point to 64-bit unsigned integer conversion), with the C rounding semantics
// (rounding towards zero), for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"
#include "endian.h"

  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__fixunssfdi)
  push {r4, lr}
  vmov r0, s0
  bl __aeabi_f2ulz
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__fixunssfdi, __aeabi_f2ulz)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_f2ulz)

  // The fast path: deliver an answer as quickly as possible for positive
  // inputs that don't overflow, and branch out of line to handle everything
  // else (negative numbers, overflows and NaNs) more slowly.
  //
  // The basic idea is to make a bare version of the mantissa, with its leading
  // 1 bit explicit at the top of the word, and shift it right by an amount
  // derived from the exponent.

  // Shift the exponent down to the bottom of the word.
  lsrs    r2, r0, #23           // r2 = exponent, or >=256 if input is negative
  rsbs    r2, r2, #63 + 0x7f    // r2 = how much to shift the mantissa right

  // If the shift count is negative, that means the input is too big, or the
  // exponent is 0xFF (so we might have a NaN), or the sign bit was set (so the
  // input is negative). Branch out of line to handle all those cases.
  blo     LOCAL_LABEL(uncommon)

  // Shift the mantissa to the top of the word, and put on the leading 1 bit.
  lsl     r3, r0, #8
  orr     r3, r3, #0x80000000

  // Make the top word of the result, which is the easy part: if the shift
  // count is too big, nothing goes wrong, we just end up with whatever part of
  // the mantissa remained in this word.
  mov     xh, r3, lsr r2

  // Make the bottom word of the result. This might involve shifting the
  // mantissa either left or right, depending on the exponent.
  //
  // We're rounding towards zero, so bits shifted off the bottom can just be
  // ignored.
  subs    r12, r2, #32          // r12 = how far to shift mantissa down
  movhs   xl, r3, lsr r12       // if that's positive, just do it
  rsblo   r12, r12, #0          // otherwise, negate it
  movlo   xl, r3, lsl r12       // and shift left by that much instead
  bx      lr

LOCAL_LABEL(uncommon):
  // We come here for positive overflows, positive infinity, NaNs, and anything
  // with the sign bit set.
  //
  // Start by testing the sign bit. Branch out of line for positive values.
  tst     r0, r0
  bpl     LOCAL_LABEL(positive_invalid)

  // Now we have either a negative finite value, -inf, or a NaN (with the sign
  // bit set). Conveniently, _all_ of those just return 0, so we don't have to
  // bother checking which.
  //
  // If we were reporting exceptions, then inputs in the range (-1,0] would
  // _legally_ return 0 (after rounding toward zero), whereas -1 and below
  // would raise the IEEE Invalid Operation exception.
LOCAL_LABEL(return_zero):
  mov     xl, #0
  mov     xh, #0
  bx      lr

LOCAL_LABEL(positive_invalid):
  // Here we have a positive value causing an invalid operation exception. If
  // it's a NaN, we return zero; otherwise we return UINT_MAX.
  mov     r1, #0xFF000000
  cmp     r1, r0, lsl #1
  blo     LOCAL_LABEL(return_zero)
  mov     xl, #0xFFFFFFFF
  mov     xh, #0xFFFFFFFF
  bx      lr

END_COMPILERRT_FUNCTION(__aeabi_f2ulz)

NO_EXEC_STACK_DIRECTIVE

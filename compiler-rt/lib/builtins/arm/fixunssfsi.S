//===-- fixunssfsi.S - single-precision FP to 32-bit unsigned int conversion==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __fixunssfsi function (single precision floating
// point to 32-bit unsigned integer conversion), with the C rounding semantics
// (rounding towards zero), for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"


  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__fixunssfsi)
  push {r4, lr}
  vmov r0, s0
  bl __aeabi_f2uiz
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__fixunssfsi, __aeabi_f2uiz)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_f2uiz)

  // The fast path: deliver an answer as quickly as possible for positive
  // inputs that don't overflow, and branch out of line to handle everything
  // else (negative numbers, overflows and NaNs) more slowly.
  //
  // The basic idea is to make a bare version of the mantissa, with its leading
  // 1 bit explicit at the top of the word, and shift it right by an amount
  // derived from the exponent.

  lsrs    r1, r0, #23           // r1 = exponent, or >=256 if input is negative
  rsbs    r2, r1, #31 + 0x7f    // r2 = how much to shift the mantissa right

  // If the shift count is negative, that means the input is too big, or the
  // exponent is 0xFF (so we might have a NaN), or the sign bit was set (so the
  // input is negative). Branch out of line to handle all those cases.
  blt     LOCAL_LABEL(uncommon)      // negative, infinite or NaN

  // Shift the mantissa up to the top of the word, OR in the leading 1, and
  // then shift it back down to make the result.
  //
  // We're rounding towards zero, so bits shifted off the bottom can just be
  // ignored.
  lsl     r3, r0, #8            // shift mantissa up
  orr     r3, r3, #0x80000000   // add leading 1
  mov     r0, r3, lsr r2        // shift down again
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
  mov     r0, #0
  bx      lr

LOCAL_LABEL(positive_invalid):
  // Here we have a positive value causing an invalid operation exception. If
  // it's a NaN, we return zero; otherwise we return UINT_MAX.
  mov     r1, #0xFF000000
  cmp     r1, r0, lsl #1
  blo     LOCAL_LABEL(return_zero)
  mov     r0, #0xFFFFFFFF
  bx      lr

END_COMPILERRT_FUNCTION(__aeabi_f2uiz)

NO_EXEC_STACK_DIRECTIVE

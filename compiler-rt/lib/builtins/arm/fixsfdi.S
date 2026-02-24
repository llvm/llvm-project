//===-- fixsfdi.S - single-precision FP to 64-bit signed int conversion ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __fixsfdi function (single precision floating point
// to 64-bit signed integer conversion), with the C rounding semantics
// (rounding towards zero), for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"
#include "endian.h"

  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__fixsfdi)
  push {r4, lr}
  vmov r0, s0
  bl __aeabi_f2lz
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__fixsfdi, __aeabi_f2lz)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_f2lz)

  // The fast path: deliver an answer as quickly as possible for positive cases
  // that don't overflow, and branch out of line to handle everything else more
  // slowly, including negative numbers, overflows, and NaNs.
  //
  // The basic idea is to make a bare version of the mantissa, with its leading
  // 1 bit explicit at the top of the word, and shift it right by an amount
  // derived from the exponent.

  // Shift the exponent down to the bottom of the word. Using ASR here in place
  // of LSR means the sign bit keeps its original value, and therefore as a
  // side effect the N flag will tell us whether the input was negative. On the
  // positive path the output is the same anyway.
  asrs    r2, r0, #23

  // Shift the mantissa to the top of the word, and put on the leading 1 bit.
  mov     r3, r0, lsl #8
  orrne   r3, r3, #1 << 31

  // If the value was negative, branch out of line to handle that.
  bmi     LOCAL_LABEL(negative)

  // Convert the mantissa into a shift count. If that's negative or zero, that
  // means the input is too big, or the exponent is 0xFF (so we might have a
  // NaN), so branch out of line again.
  rsbs    r2, r2, #63 + 0x7f
  bls     LOCAL_LABEL(invalid)

  // Make the top word of the result, which is the easy part: if the shift
  // count is too big, nothing goes wrong, we just end up with whatever part of
  // the mantissa remained in this word.
  mov     xh, r3, lsr r2

  // Make the bottom word of the result. This might involve shifting the
  // mantissa either left or right, depending on the exponent.
  subs    r12, r2, #32          // r12 = how far to shift mantissa down
  movhs   xl, r3, lsr r12       // if that's positive, just do it
  rsblo   r12, r12, #0          // otherwise, negate it
  movlo   xl, r3, lsl r12       // and shift left by that much instead
  bx      lr

LOCAL_LABEL(negative):
  // Handle negative numbers. We come here with the mantissa already prepared
  // in r3, and the exponent in the bottom 8 bits of r2 with all 1s above it
  // (because it was shifted down via ASR and the sign bit was set).

  // Start by clearing the top 24 bits of r2, left set by the ASR above,
  // leaving just the bare exponent.
  and     r2, r2, #0xff

  // Now do exactly the same processing as on the positive path.
  rsbs    r2, r2, #63 + 0x7f   // make the shift count
  bls     LOCAL_LABEL(invalid)         // branch out of line if shift count < 0
  mov     xh, r3, lsr r2       // top word of result
  subs    r12, r2, #32         // right-shift required for bottom word
  movhs   xl, r3, lsr r12      // bottom word, if shift count >= 0
  rsblo   r12, r2, #32         // otherwise, turn into a left-shift count
  movlo   xl, r3, lsl r12      // bottom word, if it needed a left shift

  // Finally, negate the answer.
  rsbs    xl, xl, #0           // negate bottom word
#if !__thumb__
  rsc     xh, xh, #0           // negate top word
#else
  // Thumb has no RSC, so simulate it by bitwise inversion and then ADC
  mvn     xh, xh
  adc     xh, xh, #0
#endif

  bx      lr

LOCAL_LABEL(invalid):
  // We come here if the exponent field of the number is large enough that it's
  // either a NaN or infinity, or a finite number of absolute value at least
  // 2^63.
  //
  // For out-of-range positive values, we return the maximum positive signed
  // integer 0x7fffffffffffffff. For out-of-range negative values, we return
  // the minimum negative signed integer 0x8000000000000000. For NaNs, we
  // return zero.
  //
  // Not _every_ number of this kind is actually an invalid input. The exact
  // value -2^63 is perfectly valid. If this implementation supported FP
  // exceptions, we'd have to detect that one case and return
  // 0x8000000000000000 with no exception, while raising an Invalid Operation
  // exception for everything else. But since we don't support exceptions, we
  // don't have to tell the difference here: -2^63 and negative overflows both
  // return 0x8000000000000000, and it doesn't matter that one is the right
  // answer and the other a best-effort error response.
  mov     r1, #0xFF000000
  cmp     r1, r0, lsl #1         // 0xFF000000 < (input << 1) means a NaN
  blo     LOCAL_LABEL(return_zero)       // so branch out of line to return zero
  mov     r2, #0x7FFFFFFF        // set up to return INT_MAX
  eor     xh, r2, r0, asr #31    // flip top half to 80000000 if input < 0
  mvn     xl, r0, asr #31        // and bottom half is FFFFFFFF or 00000000
  bx      lr

LOCAL_LABEL(return_zero):
  mov     xl, #0
  mov     xh, #0
  bx      lr

END_COMPILERRT_FUNCTION(__aeabi_f2lz)

NO_EXEC_STACK_DIRECTIVE

//===-- fixsfsi.S - single-precision FP to 32-bit signed int conversion ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __fixsfsi function (single precision floating point
// to 32-bit signed integer conversion), with the C rounding semantics
// (rounding towards zero), for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__fixsfsi)
  push {r4, lr}
  vmov r0, s0
  bl __aeabi_f2iz
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__fixsfsi, __aeabi_f2iz)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_f2iz)

  // The fast path: deliver an answer as quickly as possible for cases that
  // don't overflow or involve a NaN, and branch out of line to handle
  // everything else more slowly.
  //
  // The basic idea is to make a bare version of the mantissa, with its leading
  // 1 bit explicit at the top of the word, and shift it right by an amount
  // derived from the exponent.

  lsls    r1, r0, #1            // r1 = everything but the sign bit
  lsrs    r2, r1, #24           // r2 = just the exponent
  rsbs    r3, r2, #31 + 0x7f    // r3 = how much to shift the mantissa right

  // If the shift count is negative, that means the input is too big, or else
  // the exponent is 0xFF (so we might have a NaN). Branch out of line to
  // handle both cases.
  bls     LOCAL_LABEL(invalid)

  // OR the leading 1 into the mantissa and shift it up to the top of the word.
  //
  // We do it in _that_ order, because that way, the OR operates on the
  // original r0 and doesn't touch the top bit. So if we make it set the flags
  // too, it also tests the sign of the input, leaving the result in the N
  // flag, saving us a separate sign test instruction.
  orrs    r2, r0, #1 << 23      // put on the leading 1 and test sign
  lsl     r2, r2, #8            // shift mantissa up to top of word

  // Now shift the mantissa down to its output position, and negate the result
  // if the input was negative.
  //
  // We're rounding towards zero, so bits shifted off the bottom can just be
  // ignored.
  lsr     r0, r2, r3            // construct the rounded-down result
  rsbmi   r0, r0, #0            // negate it if input < 0
  bx      lr                    // and return

LOCAL_LABEL(invalid):
  // We come here if the exponent field of the number is large enough that it's
  // either a NaN or infinity, or a finite number of absolute value at least
  // 2^31.
  //
  // For out-of-range positive values, we return the maximum positive signed
  // integer 0x7fffffff. For out-of-range negative values, we return the
  // minimum negative signed integer 0x80000000. For NaNs, we return zero.
  //
  // Not _every_ number of this kind is actually an invalid input. The exact
  // value -2^31 is perfectly valid. If this implementation supported FP
  // exceptions, we'd have to detect that one case and return 0x80000000 with
  // no exception, while raising an Invalid Operation exception for everything
  // else. But since we don't support exceptions, we don't have to tell the
  // difference here: -2^31 and negative overflows both return 0x80000000, and
  // it doesn't matter that one is the right answer and the other a best-effort
  // error response.
  mov     r1, #0xFF000000
  cmp     r1, r0, lsl #1         // 0xFF000000 < (input << 1) means a NaN
  blo     LOCAL_LABEL(return_zero)       // so branch out of line to return zero
  mov     r2, #0x7FFFFFFF        // set up to return INT_MAX
  eor     r0, r2, r0, asr #31    // turn it into INT_MIN if input was negative
  bx      lr

LOCAL_LABEL(return_zero):
  mov     r0, #0
  bx      lr

END_COMPILERRT_FUNCTION(__aeabi_f2iz)

NO_EXEC_STACK_DIRECTIVE

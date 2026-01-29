//===-- fixunsdfsi.S - double-precision FP to 32-bit unsigned int conversion==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __fixunsdfsi function (double precision floating
// point to 32-bit unsigned integer conversion), with the C rounding semantics
// (rounding towards zero), for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"
#include "endian.h"

  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__fixunsdfsi)
  push {r4, lr}
  VMOV_FROM_DOUBLE(r0, r1, d0)
  bl __aeabi_d2uiz
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__fixunsdfsi, __aeabi_d2uiz)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_d2uiz)

  // The fast path: deliver an answer as quickly as possible for positive
  // inputs that don't overflow, and branch out of line to handle everything
  // else (negative numbers, overflows and NaNs) more slowly.
  //
  // The basic idea is to make a bare version of the mantissa, with its leading
  // 1 bit explicit at the top of the word, and shift it right by an amount
  // derived from the exponent.

  lsrs    r3, xh, #20           // exponent, or >=0x800 if input is negative

  // Convert the exponent into a right shift count. This involves subtracting
  // from 0x41e (which is the exponent of 2^31, so the largest that doesn't
  // overflow). We can't do that in one instruction, so do it in two.
  sub     r3, r3, #0x400
  rsbs    r3, r3, #0x1e

  // That RSB instruction also set the flags, which tell us whether the shift
  // count went below zero: we subtracted an adjusted input exponent from 0x1e,
  // so the LT condition is met if 0x1e < that adjusted exponent, i.e. if the
  // shift count is negative, i.e. if the input float has absolute value at
  // least 2^32.
  //
  // We didn't bother removing the sign bit when we shifted right to get the
  // exponent. So negative numbers will be treated the same as overlarge
  // positive numbers and NaNs, i.e. will branch out of line, and we can sort
  // out which is which later.
  //
  // Now check if the shift count exceeds 0xFF, so that the AArch32 shift
  // semantics would reduce it mod 0x100. We can do that in a way that also
  // sets the GT condition, saving an instruction on the fast path at the cost
  // of having to re-test which condition happened once we branch out of line.
  rsbsge  r12, r3, #0xFF        // also set LT if shift count > 0xFF
  blt     LOCAL_LABEL(uncommon)

  // Construct a word containing the top 32 bits of the mantissa.
  orr     r2, xh, #1 << 20      // put on the leading 1
  lsls    r2, r2, #11           // shift mantissa part of xh up to top of word
  orr     r2, r2, xl, lsr #21   // the top 11 mantissa bits from xl

  // Now shift the mantissa down to its output position, moving it into the
  // output register r0 in the process.
  //
  // We're rounding towards zero, so bits shifted off the bottom can just be
  // ignored.
  lsrs    r0, r2, r3            // construct the rounded-down result
  bx      lr                    // and return

LOCAL_LABEL(uncommon):
  // We come here if the exponent field of the number is either too large or
  // too small, or if the sign bit is set, so that the number is one of
  //  - a NaN
  //  - an infinity
  //  - a positive finite number of absolute value at least 2^31
  //  - a positive finite nonzero number small enough to underflow to zero (and
  //    so small that the shift instruction in the fast path couldn't handle
  //    it)
  //  - zero
  //  - any negative number at all.

  // Check for a too-large shift count (too-small exponent) first, because that
  // includes the case of an exact zero, which is probably one of the more
  // common inputs and should be handled fast if we can.
  cmp     r3, #0xFF
  movgt   r0, #0
  bxgt    lr

  // Next test the sign bit. Branch out of line for positive values, again on
  // the theory that reasonably sensible negative values (like -0.5) are more
  // likely than any of the remaining positive cases.
  tst     xh, xh
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
  lsls    r2, xh, #1              // shift exponent of x to top of word
  cmn     r2, #1 << 21            // set HI if xh proves it's a NaN by itself
  cmpeq   xl, #0                  // now HI is set if it's a NaN
  bhi     LOCAL_LABEL(return_zero)
  mov     r0, #0xFFFFFFFF
  bx      lr

END_COMPILERRT_FUNCTION(__aeabi_d2uiz)

NO_EXEC_STACK_DIRECTIVE

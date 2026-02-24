//===-- fixdfsi.S - double-precision FP to 32-bit signed int conversion ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __fixdfsi function (double precision floating point
// to 32-bit signed integer conversion), with the C rounding semantics
// (rounding towards zero), for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"
#include "endian.h"

  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__fixdfsi)
  push {r4, lr}
  VMOV_FROM_DOUBLE(r0, r1, d0)
  bl __aeabi_d2iz
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__fixdfsi, __aeabi_d2iz)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_d2iz)

  // The fast path: deliver an answer as quickly as possible for cases that
  // don't overflow or involve a NaN, and branch out of line to handle
  // everything else more slowly.
  //
  // The basic idea is to make a bare version of the top 32 bits of the
  // mantissa, with its leading 1 bit explicit at the top of the word, and
  // shift it right by an amount derived from the exponent.

  // Separate out the exponent of the input.
  lsls    r3, xh, #1            // everything but the sign bit
  lsrs    r3, r3, #21           // just the exponent

  // Convert the exponent into a right shift count. This involves subtracting
  // from 0x41e (which is the exponent of 2^31, so the largest that doesn't
  // overflow). We can't do that in one instruction, so do it in two.
  sub     r3, r3, #0x400
  rsbs    r3, r3, #0x1e

  // That RSB instruction also set the flags, which tell us whether the shift
  // count went below zero: we subtracted an adjusted input exponent from 0x1e,
  // so the LE condition is met if 0x1e ≤ that adjusted exponent, i.e. if the
  // shift count is negative or zero, i.e. if the input float has absolute
  // value at least 2^31.
  //
  // Now check if the shift count exceeds 0xFF, so that the AArch32 shift
  // semantics would reduce it mod 0x100. We can do that in a way that also
  // sets the GT condition, saving an instruction on the fast path at the cost
  // of having to re-test which condition happened once we branch out of line.
  rsbsgt  r12, r3, #0x100       // also set LE if shift count ≥ 0x100
  ble     LOCAL_LABEL(uncommon)

  // Construct a word containing the top 32 bits of the mantissa. We do this in
  // such a way that the initial ORRS also sets the N flag based on the sign
  // bit in xh, so that we can use that for a conditional negation later.
  orrs    r2, xh, #1 << 20      // put on the leading 1 and test sign
  lsl     r2, r2, #11           // shift mantissa part of xh up to top of word
  orr     r2, r2, xl, lsr #21   // the top 11 mantissa bits from xl

  // Now shift the mantissa down to its output position, moving it into the
  // output register r0 in the process, and negate the result if the input was
  // negative.
  //
  // We're rounding towards zero, so bits shifted off the bottom can just be
  // ignored.
  lsr     r0, r2, r3            // construct the rounded-down result
  rsbmi   r0, r0, #0            // negate it if input < 0
  bx      lr                    // and return

LOCAL_LABEL(uncommon):
  // We come here if the exponent field of the number is either too large or
  // too small, so that the number is one of
  //  - a NaN
  //  - an infinity
  //  - a finite number of absolute value at least 2^31
  //  - a finite nonzero number small enough to underflow to zero (and so small
  //    that the shift instruction in the fast path couldn't handle it)
  //  - zero.
  //
  // For out-of-range positive values, we return the maximum positive signed
  // integer 0x7fffffff. For out-of-range negative values, we return the
  // minimum negative signed integer 0x80000000. For everything else - NaNs,
  // underflows and true zero inputs - we return zero.
  //
  // (Not _every_ out-of-range finite number is actually an invalid input. The
  // exact value -2^31 is perfectly valid. If this implementation supported FP
  // exceptions, we'd have to detect that one case and return 0x80000000 with
  // no exception, while raising an Invalid Operation exception for everything
  // else. But since we don't support exceptions, we don't have to tell the
  // difference here: -2^31 and negative overflows both return 0x80000000, and
  // it doesn't matter that one is the right answer and the other a best-effort
  // error response.)

  // Check for a too-large shift count (too-small exponent) first, because that
  // includes the case of an exact zero, which is probably one of the more
  // common inputs and should be handled fast if we can.
  cmp     r3, #0xFF
  movgt   r0, #0
  bxgt    lr

  // Now check for NaNs.
  lsls    r2, xh, #1              // shift exponent of x to top of word
  cmn     r2, #1 << 21            // set HI if xh proves it's a NaN by itself
  cmpeq   xl, #0                  // now HI is set if it's a NaN
  bhi     LOCAL_LABEL(nan)

  // The remaining case is a too-large exponent. Return either INT_MAX or
  // INT_MIN depending on sign.
  mov     r2, #0x7FFFFFFF         // start with INT_MAX
  eor     r0, r2, xh, asr #31     // flip all its bits if xh bit 31 is set
  bx      lr

LOCAL_LABEL(nan):
  // Out-of-line path that returns zero for NaN inputs.
  mov     r0, #0
  bx      lr

END_COMPILERRT_FUNCTION(__aeabi_d2iz)

NO_EXEC_STACK_DIRECTIVE

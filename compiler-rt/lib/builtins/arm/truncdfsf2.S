//===-- truncdfsf2.S - double- to single precision FP conversion ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __truncdfsf2 function (double to single precision
// floating point conversion), with the IEEE-754 default rounding (to nearest,
// ties to even), for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"
#include "endian.h"

  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__truncdfsf2)
  push {r4, lr}
  VMOV_FROM_DOUBLE(r0, r1, d0)
  bl __aeabi_d2f
  vmov s0, r0
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__truncdfsf2, __aeabi_d2f)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_d2f)

  // Start with the fast path, dealing with input values that give a normalized
  // single-precision output. We handle these as quickly as possible in
  // straight-line code, and branch out of line to a single 'handle everything
  // else' label which will have to figure out what kind of unusual thing has
  // happened.

  // Split xh into the sign bit (in r3) and everything else (r2), so that we
  // can change the width of the exponent field and then put the sign back on
  // later.
  bic     r2, xh, #0x80000000
  and     r3, xh, #0x80000000

  // Rebias the exponent, still in its double-precision location, to account
  // for the difference between double- and single-precision exponents.
  sub     r2, r2, #(0x3ff-0x7f) << 20

  // If the exponent field is now 0 or less, we have an underflow or an exact
  // zero. If it's 0xFF or more, we have an overflow, or a NaN or infinity as
  // input. Detect all of those in a combined test, and branch out of line.
  cmp     r2, #0x00100000       // LO if output too small
  rsbshs  r12, r2, #0x0ff00000  // otherwise, set LS if output too large
  bls     LOCAL_LABEL(uncommon)          // so now LS means one or the other happened

  // We've disposed of all the uncommon cases, so we know we're returning a
  // normalized float, but we might still need to round it. Shift the round bit
  // into the C flag, also setting Z if everything below that is zero.
  lsls    r12, xl, #4

  // Put the result back together, by recombining the sign (in r3) with the
  // exponent and top of the mantissa (in r2, needing to be shifted left 3
  // bits), plus the top 3 bits of xl. The last of those is put on with an ADC
  // instruction, which also rounds up if the bit we just shifted into C was
  // set.
  orr     r2, r3, r2, lsl #3    // sign + exponent + most of mantissa
  adc     r0, r2, xl, lsr #29   // low 3 bits of mantissa + maybe round up

  // If C=1 and Z=1, we need to round to even. Otherwise we're finished. So we
  // conditionally return based on one of those flags, then clear the low
  // output bit based on the other.
  //
  // Which way round? On the assumption that input mantissas are roughly
  // uniformly distributed, _almost all_ input doubles will contain a 1 bit
  // somewhere in the bottom 28 bits, so we return early in the vast majority
  // of cases by testing Z first. If we tested C first, we'd expect to return
  // early only half the time, costing two extra instructions half the time
  // instead of 1/2^28 of the time.
  //
  // (That's a bit optimistic, because of course in some situations input
  // mantissas _won't_ be that uniform. In particular, if you converted from a
  // float, did a small amount of calculation in double, and converted back,
  // the round-to-even case might come up more often. But at least _some_
  // applications will be passing doubles that make use of the whole mantissa,
  // so I think this is still the more sensible way round to do the test.)
  bxne    lr                    // return if Z=0
  biccs   r0, r0, #1            // Z=1, so round to even if C=1 too
  bx      lr                    // and now return unconditionally

LOCAL_LABEL(uncommon):
  // We come here if anything at all goes wrong on the fast path. We could have
  // an interesting kind of input - zero, denormal, infinity or NaN - or we
  // could have a normalized double-precision input too large or too small to
  // yield a normalized single-precision output.
  //
  // Of the various cases, the most important one to handle quickly is a zero
  // input, because those are probably fairly common. So the very first thing
  // we do is test if the input is zero, and if so, return the same sign of
  // zero by simply using xh as the return value.
  orrs    r12, xl, xh, lsl #1   // are all bits of xh:xl 0 except the sign bit?

#ifndef __BIG_ENDIAN__
  // In little-endian, xh (containing the desired sign bit) and r0 (the output
  // register) aren't the same. This instruction can be skipped in big-endian,
  // where the correct output value is already in r0.
  moveq   r0, xh
#endif
  bxeq    lr

  // Separate the remaining cases into three types: too small (underflow,
  // whether or not the input was a denormal), too big (overflow or input
  // infinity, which we treat the same in the absence of FP exceptions), and
  // NaN.
  //
  // At this stage r2 contains the output exponent, rebiased to its
  // single-precision value, but at bit 20 (that is, still in the
  // double-precision position). Detect underflow by doing a signed comparison
  // against the minimum normalized single-precision exponent.
  cmp     r2, #0x00100000
  blt     LOCAL_LABEL(underflow)

  // Now figure out whether we had a NaN as input, by shifting xh left by a bit
  // (discarding the sign) and setting the new low bit if xl != 0. This gives a
  // value which is greater than 0xFFE00000 (in an unsigned comparison) for
  // precisely NaN inputs.
  cmp     xl, #1                // set C if xl != 0
  adc     r12, xh, xh           // shift that in to the bottom of xh
  cmn     r12, #0x00200000      // is the result > 0xFFE00000?
  bhi     LOCAL_LABEL(nan)               // if so, go and handle a NaN

  // If we're still here, we have a finite overflow, or an input infinity. We
  // don't have to figure out which: we return an infinity of the appropriate
  // sign in both cases. So keep just the sign of xh, and make an infinity out
  // of the rest of the bits.
  mvn     r0, xh, lsr #31       // shift sign bit down to bit 0 and flip it
  mvn     r0, r0, lsl #8        // flip it back, putting 8 set bits below it
  lsl     r0, r0, #23           // and shift those 9 bits back up to the top
  bx      lr

LOCAL_LABEL(nan):
  // We have a double-precision NaN input. The Arm NaN handling rules say that
  // we make the output single-precision NaN by keeping the sign and as much of
  // the mantissa as possible (starting from the top bit). But we also set the
  // top bit of the mantissa, which makes the output NaN quiet even if the
  // input one was signaling.
  //
  // So this code looks a bit like a miniature version of the fast path: we
  // keep the bottom 8 bits of the exponent in xh as the output exponent (we
  // know it's all 1s, which is what we want), plus all the mantissa bits below
  // it; shift all of that 3 bits left and recombine with the sign; then
  // combine with the top 3 bits of xl. Finally, set the top mantissa bit.
  bic     r2, xh, #0xF0000000   // everything from xh we want to shift left
  orr     r0, r3, xl, lsr #29   // combine sign with low 3 output mantissa bits
  orr     r0, r0, r2, lsl #3    // combine that with the shifted-up value in r2
  orr     r0, r0, #0x00400000   // set the top mantissa bit to make it a QNaN
  bx      lr

LOCAL_LABEL(underflow):
  // We have an input value small enough to underflow. The basic strategy is to
  // leave __funder to deal with the details.
  //
  // Normally __funder expects to get a value that's already been rounded, and
  // will re-round it, for which it also needs to know which way the value has
  // been rounded already. In this case we haven't rounded _yet_. Rather than
  // carefully rounding to nearest, it's easier to just make the __funder input
  // value by truncating the mantissa (i.e. round towards zero), and set the
  // rounding direction accordingly.

  // Rebias the exponent (again) to make an IEEE 754 underflow intermediate. If
  // this still doesn't make r2 positive, then the result is so small that it
  // will underflow to 0 anyway, so it doesn't really matter what exponent we
  // do provide - we just clear the top 8 bits of r2 to ensure the sign is
  // right and the exponent is _something_ small.
  adds    r2, r2, #0x0c000000   // exponent bias (still shifted down 3 bits)
  bicmi   r2, r2, #0xff000000   // handle exponent still being negative

  // Test the bits we're going to shift off the mantissa, to see if any are
  // zero. This will determine the rounding direction we pass to __funder,
  // because although we never round _up_ on this path, we must still tell it
  // whether the value we pass it was rounded down or was already exact.
  lsls    r12, xl, #3           // set Z if the intermediate value is exact

  // Put together the intermediate value to pass to __funder.
  orr     r2, r3, r2, lsl #3    // sign + exponent + most of mantissa
  orr     r0, r2, xl, lsr #29   // combine with top 3 bits of xl

  // Set the rounding direction flag based on the test above.
  moveq   r1, #0                // intermediate is exact
  movne   r1, #1                // intermediate is too small (we didn't round)

  // And tailcall __funder to do the rest of the job.
  b       SYMBOL_NAME(__compiler_rt_funder)

END_COMPILERRT_FUNCTION(__aeabi_d2f)

NO_EXEC_STACK_DIRECTIVE

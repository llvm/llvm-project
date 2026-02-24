//===-- divdf3.S - double-precision floating point division ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __divdf3 function (double precision floating point
// division), with the IEEE-754 default rounding (to nearest, ties to even),
// for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"
#include "endian.h"

// The basic strategy of this division code is to use Newton-Raphson iteration
// to calculate an approximation to 1/y, then multiply it by x. This procedure
// delivers a quotient with 10 extra bits of precision, but which isn't exact.
// We know an upper bound on its possible error, which gives an interval of
// possible values for the true quotient. So we can check the 10 extra bits to
// see whether a rounding boundary lies within the interval. If not, then we
// can round and return without worrying further; otherwise, we go to slower
// correction code that multiplies the approximate quotient back up by y and
// checks it against x.
//
// This strategy depends critically on the upper bound on the approximation
// error. Underestimating the error introduces a bug; overestimating it costs
// performance, by sending more cases than necessary to the slow path.
//
// To give high confidence of its correctness, the upper bound has been proved
// formally by Gappa. The Gappa proof and auxiliary code are not included in
// this version, but they can be found in the Arm Optimized Routines repository
//
// https://github.com/ARM-software/optimized-routines/blob/bf3e44c3784dd3e18d3d5232e13b4d81f232310b/fp/at32/ddiv.S
// https://github.com/ARM-software/optimized-routines/blob/bf3e44c3784dd3e18d3d5232e13b4d81f232310b/fp/auxiliary/ddiv-prove.py
// https://github.com/ARM-software/optimized-routines/blob/bf3e44c3784dd3e18d3d5232e13b4d81f232310b/fp/auxiliary/ddiv-diagnostics.c
//
// and a pair of blog posts describing the concepts and procedure are here:
//
// https://developer.arm.com/community/arm-community-blogs/b/embedded-and-microcontrollers-blog/posts/formally-verifying-a-floating-point-division-routine-with-gappa-p1
// https://developer.arm.com/community/arm-community-blogs/b/embedded-and-microcontrollers-blog/posts/formally-verifying-a-floating-point-division-routine-with-gappa-p2

  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__divdf3)
  push {r4, lr}
  VMOV_FROM_DOUBLE(r0, r1, d0)
  VMOV_FROM_DOUBLE(r2, r3, d1)
  bl __aeabi_ddiv
  VMOV_TO_DOUBLE(d0, r0, r1)
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__divdf3, __aeabi_ddiv)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_ddiv)

  push    {r4,r5,r6,r7,r8,lr}

  // Check if either input exponent 7FF (infinity or NaN), and if so, branch
  // out of line.
  ldr     r12, =0x07FF0000        // mask for exponent cold storage
  bics    r4, r12, xh, lsr #4     // test for Infs or NaNs
  bicsne  r4, r12, yh, lsr #4
  beq     LOCAL_LABEL(ddiv_naninf)

  // Extract the exponents of the input values x and y into bits 16..26 of r14
  // and r5 respectively, and in the process, check if either exponent is zero
  // (so that one or both inputs are 0 or denormal). In order to combine the
  // two tests, the second ANDS is performed conditionally, so that if x's
  // exponent is zero then the out-of-line code at ddiv_zerodenorm might find
  // y's exponent hasn't been set up yet.
  //
  // We also calculate the sign of the result, which will be needed whether or
  // not we branch. This is saved in the low bit of r4.
  ands    r4, r12, xh, lsr #4     // get exponent of x, setting Z if it's 0
  andsne  r5, r12, yh, lsr #4     // if not, extract and test exponent of y
  eor     r6, xh, yh              // XOR the input signs to get the result sign
  orr     r4, r4, r6, lsr #31     // save it in the low bit of r4
  beq     LOCAL_LABEL(ddiv_zerodenorm)         // branch out of line for zeroes or denormals

  // Calculate the initial exponent of the result, by subtracting the two input
  // exponents and adjusting for the IEEE exponent bias. This value may have to
  // be adjusted by 1 later, depending on the quotient of the mantissas.
  //
  // If we branched to ddiv_zerodenorm above, and it found denormals but no
  // zeroes, it may branch back here after renormalising them. We expect the
  // out-of-line code to have left the exponent difference in the top half of
  // r4 (still with the output sign in the low bit), but not yet to have
  // applied the bias. So it branches back in immediately after the SUB.
  //
  // The exponent bias we want is either 0x3fe or 0x3ff, depending on whether
  // we have to shift the output mantissa by 1 below. Neither of those values
  // fits in the immediate field of an ADD instruction, so we must use two
  // instructions.
  sub     r4, r4, r5
LOCAL_LABEL(ddiv_normalised): // denormal handler will come back to here
  add     r4, r4, #0x03FC0000     // add the 8 high bits of the bias 0x3FE
  add     r4, r4, #0x00020000     // add the remaining bit of the bias

  // Shift both mantissas up to the top of their 64-bit register pair, and OR
  // in the leading 1 bit, which will occupy the high bit of the high word in
  // each case.
  mov     r5, #1<<31              // high bit for ORing in to both mantissas
  orr     xh, r5, xh, lsl #11     // shift up xh and OR in the high bit
  orr     yh, r5, yh, lsl #11     // same for yh
  orr     xh, xh, xl, lsr #21     // OR in the bits shifted out of xl into xh
  orr     yh, yh, yl, lsr #21     // same for yl and yh
  lsl     xl, xl, #11             // shift up the rest of xl
  lsl     yl, yl, #11             // same for yl

  // Check if the two mantissas are exactly equal, so that the quotient is
  // exactly a power of 2. If so, branch out of line to handle that case
  // specially.
  //
  // This guarantees that when we examine the approximate quotient afterwards,
  // we can't be confused about whether it needs to be renormalised, which
  // would otherwise cost just as much effort as this check. Our reciprocal
  // approximation is always an underestimate (that's in the nature of this
  // particular Newton-Raphson iteration), so if x < y (meaning the mantissas
  // rather than the whole floats) then even the true quotient will be less
  // than 1, and the approximation even more so. On the other hand, if x > y,
  // then the true quotient will be enough greater than 1 that even the largest
  // possible error in the approximation can't make it look like less than 1.
  //
  // (Proof: regard x,y as normalised to the range [1,2). If x > y, then we
  // have x ≥ y+ε, where ε is the machine epsilon. So x/y ≥ 1+ε/y > 1+ε/2. And
  // the bound on the approximation error, given below, is far less than ε/2.)
  cmp     xh, yh
  cmpeq   xl, yl
  beq     LOCAL_LABEL(ddiv_result_is_power_of_2)

  // Now we begin the actual calculation of the reciprocal approximation.
  //
  // We begin with our two input mantissas stored in xh:xl and yh:yl, each with
  // its leading 1 explicit and shifted up to the top of the word. So they can
  // be regarded as 64-bit integers with the high bit set and the bottom 11
  // bits clear.

  // Obtain an 8-bit reciprocal approximation by using the topmost 8 bits of y
  // as a lookup table. The top bit of y is always set, so there are only 128
  // lookup table entries, not 256. The 8-bit value we load also has its top
  // bit set.
  lsr     r5, yh, #24           // r5 is the table index plus 0x80
  adr     r6, LOCAL_LABEL(reciptbl)-128      // so subtract 0x80 from the table address
  ldrb    r6, [r6, r5]          // and load the approximation into r6

  // First Newton-Raphson iteration, which expands that 8-bit approximation to
  // a 17-bit one, again with its top bit set. We use the top 16 bits of y for
  // this, so that we can fit the multiplications into ordinary MUL rather than
  // UMULL.
  //
  // The Newton-Raphson formula to turn an approximation r ≈ 1/y into a better
  // one is r → r(2-yr). In this case we're scaling up to integers (informal
  // fixed point), so the 2 becomes 2^24.
  lsr     r5, yh, #16           // get top halfword of y
  mul     r7, r6, r5            // multiply it by the input value r
  rsb     r7, r7, #1<<24        // subtract from 2 (scaled up appropriately)
  mul     r7, r6, r7            // multiply again to make r(2-yr)
  lsr     r7, r7, #14           // shift down to keep only 17 bits of it

  // Second iteration, expanding into a 32-bit reciprocal, using the top 31
  // bits of y (i.e. yh shifted by 1). The first multiplication (making yr) is
  // 32x32 → 64 bits, so we use a single UMULL; the second one making r(2-yr)
  // is 32x64, which we do with a UMULL by the bottom half of yr and then MLA
  // by the top half, so we only keep the low 64 bits of the full answer.
  //
  // The subtraction from 2 (again scaled up, this time to 2^48) is done by
  // RSBS+RSC, interleaved with the multiplications so as to use a delay slot
  // on CPUs that have one.
  lsr     r12, yh, #1
  umull   r6, r8, r7, r12       // r8:r6 = yr
  rsbs    r6, r6, #0            // low half of subtraction from 2
  umull   r12, lr, r7, r6       // multiply r by the low half of 2-yr
#if !__thumb__
  rsc     r8, r8, #1<<16        // high half of subtraction from 2
#else
  // Thumb has no RSC, so simulate it by bitwise inversion and then ADC
  mvn     r8, r8
  adc     r8, r8, #1<<16
#endif
  mla     r6, r7, r8, lr        // multiply r by the high half of 2-yr

  // Third iteration, expanding into a 64-bit reciprocal, with the leading bit
  // expected to end up in bit 60. Now the first multiplication to make yr is
  // 32x64 → 96 bits, so we put the product in three registers lr:r12:r8.
  // However, we're going to discard the low word r8 completely, because it
  // makes negligible difference. So we'll treat the output yr as 64-bit.
  umull   r8, r12, r6, yl       // multiply r by bottom half of y
  mov     lr, #0                // initialize high word to 0
  umlal   r12, lr, r6, yh       // multiply r by top half of y
  // Subtract from a power of 2, as usual. But in this case the power of 2
  // we're subtracting from is 2^64, which is just off the top of the 64-bit
  // value in lr:r12. So in fact we're just negating the whole thing!
  //
  // To preserve the invariant that the approximation error is always negative,
  // we negate via one's complement rather than two's. (This would only make a
  // difference if r8 had happened to be exactly 0. That in turn can occur when
  // yl=0, so one of the test cases in ddiv-diagnostics.c deliberately uses
  // such a value, so that the intermediate results can be checked against the
  // reference Python.)
  mvn     r12, r12
  mvn     lr, lr
  // Now lr:r12:r8 contains 2-yr. We discard the low word r8 to reduce that to
  // 64 bits, and do another 32x64 → 96 bit multiplication.
  umull   r5, r8, r6, r12      // multiply r by bottom half of 2-yr
  mov     r7, #0               // initialize high word to 0
  umlal   r8, r7, r6, lr       // multiply r by top half of 2-yr

  // That's the Newton-Raphson iteration done: we have a 64-bit approximation
  // to 1/y. Multiply it by x to get the full approximate quotient.
  //
  // In principle, this would be a 64x64 → 128 bit multiplication, involving
  // four long multiply instructions. But we only need the top 64 bits, and
  // we're already prepared to tolerate some error in the calculations, so we
  // cut corners: don't multiply the two low words together at all, and we
  // discard the bottom half of each of the (low * high) partial products
  // without bothering to propagate carries out of it.
  //
  // (All of these shortcuts are faithfully mimicked in the Python reference
  // implementation which generates Gappa input, so they're all accounted for
  // in the error analysis.)
#if __ARM_FEATURE_DSP
  umull   r12, r6, xh, r8      // r6 = high word of x * low word of 1/y
  umull   r12, r5, xl, r7      // r5 = low word of x * high word of 1/y
  umaal   r6, r5, xh, r7       // add those to the product of both high words
#else
  // Alternative instruction sequence using UMLAL, if UMAAL isn't available
  umull   r12, r6, xh, r8      // r6 = high word of x * low word of 1/y
  umull   r12, lr, xl, r7      // lr = low word of x * high word of 1/y
  adds    r6, r6, lr           // add those together
  mov     r5, #0               // set r5 to the carry out of that addition
  adc     r5, r5, #0
  umlal   r6, r5, xh, r7       // add that to the product of both high words
#endif
  // Now r5:r6 is the completed approximate quotient, with its leading bit at
  // position either 61 or 62.

  // Normalize so that the leading bit is always in bit 60, by shifting left if
  // it isn't there already, and adjusting the output exponent by 1 to
  // compensate.
  //
  // We do the test in a slightly tricky way, by arranging to set the V flag if
  // the leading bit is in bit 60. This allows us to do the left shift under
  // the VC condition, which is convenient because the LSLS instruction that
  // shifts the low word left moves the top bit into the C flag without
  // affecting V.
  //
  // We also save the value written into lr by the initial ADDS instruction,
  // because that contains enough information to tell us whether we
  // renormalised here. The correction path for quotients too close to a
  // rounding boundary will need to recover that information.
  adds    lr, r5, #0x40000000  // set V flag if bit 62 of high word set
  subvc   r4, r4, #1<<16       // if not, correct the exponent by 1,
  lslsvc  r6, r6, #1           // shift the low word of the quotient left
  adcvc   r5, r5, r5           // and shift its top bit into the high word

  // Now r5:r6 is the _normalised_ approximate quotient, with its leading bit
  // reliably in bit 60. This is the final output of the calculation that the
  // Gappa error-analysis proof applies to.

  // That 64-bit output has bit 63 clear; the leading 1 bit of the output
  // mantissa in bit 62, followed by 52 more mantissa bits; then 10 bits at the
  // bottom which are used for determining rounding.
  //
  // Compute the _approximately_ rounded-to-nearest output mantissa, by adding
  // half a ULP and shifting down. If we don't go to the slow path, this is the
  // correct output mantissa. (See fdiv.S for the proof that the round-to-even
  // tiebreaking case can't occur in floating-point division.)
  //
  // We keep the original version of r6, containing the ten rounding bits, so
  // that we can test it to see if we need the slow path.
  adds    r7, r6, #1<<9        // add half a ULP, copying low word into r7
  adc     r5, r5, #0           // propagate carry into high word
  lsr     r7, r7, #10          // shift low word right
  orr     r7, r7, r5, lsl #22  // combine with bits shifted out of high word
  lsr     r5, r5, #10          // shift high word right

  // Now test r6 to see whether this output mantissa can be relied on, or
  // whether the approximation landed too close to a rounding boundary.
  //
  // The maximum possible error in the approximation, taking into account the
  // initial error in each lookup table entry, the remaining mathematical error
  // introduced by stopping after this many Newton-Raphson iterations, and
  // every shortcut, right shift, truncation and discarding of a partial
  // product in the algorithm above, is always negative, and less than 64 units
  // in the last place of the 64-bit approximate quotient. That is, the true
  // quotient lies somewhere between the 64-bit integer described as "final
  // output of the calculation" above, and that plus 64.
  //
  // So if the bottom 10 bits of r6 have the value 2^9 or greater, we're safe,
  // because the true value is _larger_ than the approximation, so if the
  // approximation is already above the rounding boundary then so is the true
  // value. And if those 10 bits are (2^9-64) or less then we're also safe,
  // because even if the true value is greater by 63, it's still on the same
  // side of the rounding boundary.
  //
  // We check the error by subtracting (2^9-64), so that the dangerous values
  // of the bottom 10 bits are those in the range 0,...,63, i.e. precisely
  // those with none of bits 6,7,8,9 set.
  //
  // We also combine this test with a check for underflow, because that also
  // needs more careful handling (the mantissa must be re-rounded to a
  // different bit position, which involves knowing whether it's exact).
  // Underflow has happened if the exponent in the top half of r4 is negative
  // (it's off by 1 so that the leading mantissa bit will increment it), so we
  // test by an ASR#31 (copying the top bit of r4 into all of it) and negating.
  // That way, the output value is zero on underflow, matching the flags from
  // the other check.
  sub     r6, r6, #(1<<9)-64
  tst     r6, #0x3C0              // now EQ means we must go to the slow path
  mvnsne  r12, r4, asr #31        // also set EQ if underflow has happened
  beq     LOCAL_LABEL(ddiv_correction)         // branch out of line to do the hard bit

  // If we do go to ddiv_correction, it branches back here after the correction
  // code has finished. Either way, we expect that r5:r7 is the result
  // mantissa, with the top bit set, already in the correct position in the
  // word, and already rounded to nearest.
LOCAL_LABEL(ddiv_corrected):
  // Recombine the output mantissa with the sign and exponent.
  add     xh, r5, r4, lsl #31     // add sign bit to top word of mantissa
  bic     r12, r4, #1             // isolate exponent in top half of r4
  add     xh, xh, r12, lsl #4     // add exponent to make the final high word
  mov     xl, r7                  // move low word into the right register

  // If there's no overflow or underflow, we're done.
  //
  // We _identified_ underflow above when we went to the slow path, but having
  // done that, the slow path came back here, so we must check for it again.
  // (The only purpose of the detour was to obtain accurate information about
  // whether the quotient is exact, or needed rounding.)
  //
  // The output exponent, offset downwards by 1, is in the top half of r4. If
  // it's negative, there's an underflow; if it's too large, there's an
  // overflow. We do an approximate test for both at once via an unsigned
  // comparison against 0x7f0, using r12 (the register in which we already
  // cleared the sign bit stored at the bottom). This identifies _most_ normal
  // outputs as quickly as possible.
  //
  // 0x7f0 isn't the maximum possible known-safe exponent, but it's the largest
  // one that fits in the immediate field of CMP. We deal with the remaining
  // cases in the next few instructions.
  cmp     r12, #0x7f0 << 16
  popls   {r4,r5,r6,r7,r8,pc}

  // Now check the remaining cases more carefully.
  //
  // If r12 < 0 then we definitely have underflow. We detect overflow precisely
  // by seeing if the _final_ output exponent (in the output register xh) is
  // 0x7ff or more, by incrementing it and seeing if the sign is opposite from
  // the intended output sign.
  add     lr, xh, #1<<20          // increment the output exponent field
  teq     lr, r4, lsl #31         // set N if the sign now doesn't match r4[0]
  tstpl   r12, r12                // otherwise, set N if underflow
  poppl   {r4,r5,r6,r7,r8,pc}     // if neither, we've finished

  // If we still haven't returned, we really do have overflow or underflow, and
  // the sign of r12 tells us which.
  tst     r12, r12
  bmi     LOCAL_LABEL(ddiv_underflow)
  // For overflow, correct the sign by biasing the exponent downward, and go to
  // code that constructs an infinite return value (shared with the
  // division-by-zero handler).
  sub     xh, xh, #0x60000000
  pop     {r4,r5,r6,r7,r8,lr}     // ddiv_retinf expects no regs on the stack
  b       LOCAL_LABEL(ddiv_retinf)

LOCAL_LABEL(ddiv_correction):
  // The slow path, entered if the approximate quotient was too close to a
  // rounding boundary to trust, and also if there's a chance of underflow (so
  // that we can reliably determine the rounding direction, including whether
  // the quotient was exact).
  //
  // Regarding the input mantissas x,y and our approximate quotient q as
  // integers in [2^52,2^53), the quotient is an approximation to either
  // x*2^52/y or x*2^53/y, depending on which of x,y was larger. We know that q
  // is less than the true value of that quotient by at most a small fraction
  // of a ULP. So the correct rounded quotient is either equal to q or to q+1,
  // and we can decide which by multiplying back up by y: we want q - x*2^k/y
  // to be in the range (-1/2,+1/2) (where k = 52 or 53), which is equivalent
  // to asking if qy - x*2^k is in the range (-y/2,+y/2).
  //
  // That's a calculation we can do in integers using only addition and
  // multiplication. And we know that if q itself doesn't have that property
  // then q+1 will.

  // The mantissa of y is currently right at the top of the word, which means
  // that if the result of our check is greater than it, it will overflow. So
  // we must start by shifting y downward. We'll put it back at the bottom of
  // the word, where it was in the input float.
  lsr     yl, yl, #11             // shift yl right
  orr     yl, yl, yh, lsl #21     // OR in the bits shifted out of yh
  lsr     yh, yh, #11             // shift yh right

  // Compute the integer qy-x. Because q is already very close to the right
  // quotient, we expect this to be an integer at most twice the size of y,
  // which easily fits in 64 bits. So we don't need to compute the full 128-bit
  // product: the low 64 bits are enough.
  umull   r8, r6, r7, yl          // 64-bit product of the low words
  mla     r6, r7, yh, r6          //   + (high word of y) * (low word of q)
  mla     r6, r5, yl, r6          //   + (high word of q) * (low word of y)

  // Now we must subtract either x << 53 or x << 52. This will only affect the
  // high word of the product we've just computed. Also the mantissa of x is
  // already shifted left by 11. So we shift xl left by either (52-32-11) or
  // (53-32-11), i.e. by 9 or by 10, and subtract from the high word of the
  // product.
  //
  // To decide which, we consult the value left in lr by the original test for
  // renormalization, which added 0x40000000 to the high word of the initial
  // approximate quotient 'quot'. If that had bit 62 set (so no renormalization
  // needed) then the addition carried into the sign bit; otherwise it didn't.
  // So lr is positive if and only if we need to shift xl left by an extra bit.
  tst     lr, lr                  // did we renormalize?
  subpl   r6, r6, xl, lsl #10     // if so, subtract x<<53 from q*y
  submi   r6, r6, xl, lsl #9      // if not, subtract x<<52

  // Now r6:r8 contains the residual value r = qy - x*2^k as described above.
  // If this is between -y/2 and +y/2 then q is already the correctly rounded
  // quotient. Otherwise, the correct quotient is q+1, so the value in r6:r8
  // will be too small (incrementing q would add y to it). So we need to check
  // whether r < -y/2, or equivalently whether 2r < -y (avoiding having to
  // worry about what happens when we halve y if it's odd).
  //
  // As mentioned above, division can't give an exact halfway case, so we don't
  // need to worry about the case r = y/2.
  adds    r8, r8, r8              // multiply the residual by 2
  adc     r6, r6, r6
  adds    lr, r8, yl              // add y to it, discarding the result
  adcs    lr, r6, yh
  bpl     LOCAL_LABEL(ddiv_corrected)          // if the answer is positive, we're OK

  // If we didn't take that branch, then the approximate quotient is too small
  // by 1, so we must increment it. But also, we adjust the residual in r6:r8
  // to match. That residual is unused by the main epilogue code, but we also
  // came here for any underflowing value, and the underflow handler will need
  // the exact residual to determine the rounding direction.
  //
  // (We could re-test whether underflow had happened and use that to skip the
  // update of r6:r8, but the test would cost as much effort as it saved!)
  adds    r7, r7, #1              // increment the output quotient
  adcs    r5, r5, #0
  adds    r8, r8, yl              // repeat the addition of y to the residual,
  adcs    r6, r6, yh              //   this time keeping the result in r6:r8
  b       LOCAL_LABEL(ddiv_corrected)          // finally we can rejoin the main code

LOCAL_LABEL(ddiv_result_is_power_of_2):
  // The special-case handler for the two input mantissas being equal, so that
  // the result is an exact power of two. We set up all the output registers to
  // the way the main code would have done it, and jump straight to
  // ddiv_corrected. This includes setting r6:r8 to the 'residual' value
  // computed by the slow path, in case this power-of-2 output is also an
  // underflow, which will depend on those registers.
  mov     r5, #0x00100000         // high word of quotient mantissa = 1<<20
  mov     r7, #0                  // low word of quotient mantissa = 0
  mov     r6, #0                  // high word of residual = 0
  mov     r8, #0                  // low word of residual = 0
  b       LOCAL_LABEL(ddiv_corrected)

LOCAL_LABEL(ddiv_underflow):
  // We come here to handle underflow. The output double, constructed naïvely
  // from the out-of-range exponent, is in xh:xl. We expect in this situation
  // that we've _always_ come via either the ddiv_correction slow path or the
  // ddiv_result_is_power_of_2 special case, both of which will have set up a
  // residual value in r6:r8 equal to q*y - x*2^k (for appropriate k). This
  // value is positive if the quotient is slightly above the true value (i.e.
  // was rounded up), or negative if the quotient was rounded down. But we must
  // also distinguish the third case of the residual being exactly zero.
  add     xh, xh, #0x60000000     // apply IEEE 754 exponent bias for __dunder
  orrs    r12, r6, r8             // set r12=0 and Z=1 if quotient was exact
  movne   r12, #1                 // otherwise, set r12 = +1
  orrne   r12, r12, r6, asr #31   // and change to -1 if residual is negative
  pop     {r4,r5,r6,r7,r8,lr}     // pop all locally saved registers
  b       SYMBOL_NAME(__compiler_rt_dunder)                // and tailcall __dunder to finish

LOCAL_LABEL(ddiv_zerodenorm):
  // We come here if either input had exponent 0, so there's at least one zero
  // or denormal. However, we know there are no infinities or NaNs, because
  // those were checked first and will have gone to ddiv_naninf below.
  //
  // First we must repeat the instruction which extracted the exponent of y
  // into r5, this time unconditionally, in case the setup code didn't do it.
  and     r5, r12, yh, lsr #4

  // If either or both input is actually zero, the answer is easy.
  orrs    lr, xl, xh, lsl #1    // is x zero?
  beq     LOCAL_LABEL(ddiv_xzero)
  orrs    lr, yl, yh, lsl #1    // is y zero?
  beq     LOCAL_LABEL(ddiv_divbyzero)

  // Otherwise, delegate to __dnorm2 to handle denormals, converting them into
  // a normalised mantissa and an out-of-range exponent. __dnorm2 expects the
  // exponents at the bottom of their words instead of half way up, so shift
  // down first.
  lsr     r4, r4, #16
  lsr     r5, r5, #16
  push    {r0, r1, r2, r3, r4, r5} // create a 'struct dnorm2' on the stack
  mov     r0, sp                   // pass it by address
  bl      SYMBOL_NAME(__compiler_rt_dnorm2)
  pop     {r0, r1, r2, r3, r4, r5}

  // Rejoin the main code, with the exponent difference in the top half of r4,
  // and the output sign in the low bit of r4. (The original setup code did the
  // latter, but we clobbered it while setting up for __dnorm2.)
  subs    r4, r4, r5               // exponent difference, at the bottom of r4
  lsls    r4, r4, #16              // move it up to the right place
  orr     r4, r4, r6, lsr #31      // recover output sign from top bit of r6
  b       LOCAL_LABEL(ddiv_normalised)          // rejoin the main code

LOCAL_LABEL(ddiv_xzero):
  // We come here if x=0. We return 0 (of the right sign) if y is not 0, and
  // the default quiet NaN if both inputs are zero.
  orrs    lr, yl, yh, lsl #1       // is y zero?
  beq     LOCAL_LABEL(ddiv_ivo_pop)             // if so, pop registers and return a NaN
  // We know xl=0 already, so we only need to reset xh to contain the right
  // output sign. The setup code left that in the high bit of r6.
  and     xh, r6, #0x80000000
  pop     {r4,r5,r6,r7,r8,pc}

LOCAL_LABEL(ddiv_divbyzero):
  // We come here if y=0, but x is not 0 (or we'd have gone to ddiv_xzero above
  // instead). So we're dividing a nonzero number by zero, and must return
  // infinity.
  pop     {r4,r5,r6,r7,r8,lr}
  eor     xh, xh, yh               // combine signs to get result sign
  b       LOCAL_LABEL(ddiv_retinf)

LOCAL_LABEL(ddiv_naninf):
  // We come here knowing that at least one operand is either NaN or infinity.
  // If there's a NaN, we can tailcall __dnan2 to do the right thing. Pop our
  // stacked registers first: we won't need that much spare space any more, and
  // it makes the tailcall easier if we've already done it.
  pop     {r4,r5,r6,r7,r8,lr}

  // A number is a NaN if its exponent is 0x7ff and at least one bit below that
  // is set. The CMP + ADC pair here converts the two words xh:xl into a single
  // word containing xh shifted up by one (throwing away the sign bit which
  // makes no difference), with its low bit set if xl was nonzero. So if that
  // is strictly greater than 0xffe00000, then x was a NaN.
  cmp     xl, #1
  adc     r12, xh, xh
  cmp     r12, #0xFFE00000
  bhi     SYMBOL_NAME(__compiler_rt_dnan2)
  // Now check y in the same way.
  cmp     yl, #1
  adc     r12, yh, yh
  cmp     r12, #0xFFE00000
  bhi     SYMBOL_NAME(__compiler_rt_dnan2)

  // Now we know there are no NaNs. Therefore there's at least one infinity. If
  // both operands are infinity then we have inf / inf = invalid operation and
  // must return a NaN. We detect this by XORing the inputs' exponent fields:
  // knowing one of them is 7FF, they XOR to zero iff the other one is too.
  eors    r12, xh, yh              // XOR entire top words of the inputs
  lsl     r12, r12, #1             // shift left to discard the sign bit
  lsrs    r12, r12, #21            // shift right again to discard mantissas
  beq     LOCAL_LABEL(ddiv_ivo)                 // if what's left is 0, we have inf / inf

  // Otherwise, there's exactly one infinity, so our answers are easy, but
  // depend on which operand it is:
  //   infinity / anything = infinity
  //   anything / infinity = 0
  //
  // Determine if x is the infinity, by bitwise inverting the whole word and
  // then shifting left and right to isolate its exponent bits.
  mvn     r12, xh, lsl #1          // invert x, shift left to discard sign
  lsrs    r12, r12, #21            //   and shift right to discard mantissa
  eor     xh, xh, yh               // calculate the output sign bit
  beq     LOCAL_LABEL(ddiv_retinf)              // if x = inf, return infinity of that sign
  mov     xl, #0                   // otherwise clear all bits of x
  and     xh, xh, #0x80000000      //   other than the sign bit
  bx      lr                       //   and return zero of the same sign
LOCAL_LABEL(ddiv_retinf):
  // Construct and return an infinity in xh:xl, with whatever sign bit is
  // already in the top bit of xh.
  mov     xl, #0                   // clear low word
  mvn     xh, xh, lsr #31          // shift xh[31] down to bit 0, inverted
  mvn     xh, xh, lsl #11          // uninvert, and put exponent 0x7ff below it
  lsl     xh, xh, #20              // shift back up to the top
  bx      lr

  // Code to construct and return the default quiet NaN, for the cases inf/inf
  // and 0/0. We provide two entry labels, one for callers who still need to
  // pop all the registers this function pushed, and one for callers who have
  // done that already.
LOCAL_LABEL(ddiv_ivo_pop):
  pop     {r4,r5,r6,r7,r8,lr}
LOCAL_LABEL(ddiv_ivo):
  movw    xh, 0x7ff8
  lsls    xh, xh, #16
  mov     xl, #0
  bx      lr

  // Table of approximate reciprocals.
LOCAL_LABEL(reciptbl):
  .byte 0xFF,0xFD,0xFB,0xF9,0xF7,0xF5,0xF4,0xF2
  .byte 0xF0,0xEE,0xED,0xEB,0xE9,0xE8,0xE6,0xE4
  .byte 0xE3,0xE1,0xE0,0xDE,0xDD,0xDB,0xDA,0xD8
  .byte 0xD7,0xD5,0xD4,0xD3,0xD1,0xD0,0xCF,0xCD
  .byte 0xCC,0xCB,0xCA,0xC8,0xC7,0xC6,0xC5,0xC4
  .byte 0xC2,0xC1,0xC0,0xBF,0xBE,0xBD,0xBC,0xBB
  .byte 0xBA,0xB9,0xB8,0xB7,0xB6,0xB5,0xB4,0xB3
  .byte 0xB2,0xB1,0xB0,0xAF,0xAE,0xAD,0xAC,0xAB
  .byte 0xAA,0xA9,0xA8,0xA8,0xA7,0xA6,0xA5,0xA4
  .byte 0xA3,0xA3,0xA2,0xA1,0xA0,0x9F,0x9F,0x9E
  .byte 0x9D,0x9C,0x9C,0x9B,0x9A,0x99,0x99,0x98
  .byte 0x97,0x97,0x96,0x95,0x95,0x94,0x93,0x93
  .byte 0x92,0x91,0x91,0x90,0x8F,0x8F,0x8E,0x8E
  .byte 0x8D,0x8C,0x8C,0x8B,0x8B,0x8A,0x89,0x89
  .byte 0x88,0x88,0x87,0x87,0x86,0x85,0x85,0x84
  .byte 0x84,0x83,0x83,0x82,0x82,0x81,0x81,0x80

END_COMPILERRT_FUNCTION(__aeabi_ddiv)

NO_EXEC_STACK_DIRECTIVE

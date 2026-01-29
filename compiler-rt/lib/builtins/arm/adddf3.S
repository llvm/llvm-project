//===-- adddf3.S - Add/subtract double precision floating point numbers ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __adddf3 and __subdf3 functions (double precision
// floating point number addition and subtraction), with the IEEE-754 default
// rounding (to nearest, ties to even), for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"
#include "endian.h"

  .syntax unified
  .text
  .p2align 2

// General structure of this code:
//
// There are three actual entry points here, for addition, subtraction and
// reversed subtraction (just taking the operands the other way round, so that
// it returns y-x instead of x-y). But the first thing the functions do (after
// checking for NaNs) is to sort out whether the magnitudes of the two inputs
// are being added (x+y with like signs, or x-y with different signs), or
// subtracted. So dadd jumps across into the middle of dsub if it sees that the
// signs are different, and vice versa. Then the main code path in dadd handles
// magnitude addition, and the one in dsub handles magnitude subtraction.
//
// NaNs are checked first, so that an input NaN can be propagated exactly,
// including its sign bit. After ruling out that case, it's safe to flip the
// sign of one of the inputs, so that during the cross-calls, x - y can be
// rewritten as x + (-y) and vice versa.

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__adddf3)
  push {r4, lr}
  VMOV_FROM_DOUBLE(r0, r1, d0)
  VMOV_FROM_DOUBLE(r2, r3, d1)
  bl __aeabi_dadd
  VMOV_TO_DOUBLE(d0, r0, r1)
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__adddf3, __aeabi_dadd)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_dadd)

  push    {r4, r14}

  // Test for all uncommon values at once: infinities, NaNs, denormals and
  // zeroes. Branch out of line if any are found. We do this by incrementing
  // the exponent of each input, so that the two extreme exponents 0x7ff,0x000
  // map to 0x000,0x001 respectively. Then the original number had one of those
  // exponents precisely when the modified version has the top 10 exponent bits
  // zero.
  //
  // The constant we load into r14 for testing those ten exponent bits will be
  // reused later. (We could load a constant suitable for just this initial
  // test slightly more efficiently by writing MOVW r14,#0x3ff or similar, but
  // having the set bits at the top of the word is useful later because we can
  // extend them using ASR.)
  ldr     r14, =0xFFC00000
  add     r12, xh, #1 << 20   // r12 has the adjusted version of x's exponent
  add     r4, yh, #1 << 20    // and r4 the adjusted version of y's
  tst     r14, r12, lsl #1    // test the top 10 exponent bits of each
  tstne   r14, r4, lsl #1
  beq     LOCAL_LABEL(add_uncommon)       // and branch out of line if either is 0

  // Now we have two normalised numbers. If their signs are opposite, we should
  // be subtracting their magnitudes rather than adding, so cross-jump to dsub.
  teq     xh, yh
  eormi   yh, yh, #1 << 31
  bmi     LOCAL_LABEL(sub_magnitude)
LOCAL_LABEL(add_magnitude):
  // If we get here, we're adding operands with equal signs (i.e. a magnitude
  // addition). First thing to do is put the operands in magnitude order, so
  // that x >= y.
  subs    r4, xl, yl          // compare inputs, also keeping x-y
  sbcs    r12, xh, yh
  bhs     LOCAL_LABEL(add_swapped)        // if x>=y then branch round the swap
  adds    yl, yl, r4          // otherwise turn y into x by adding (x-y)
  adc     yh, yh, r12
  subs    xl, xl, r4          // and turn x into y by subtracting it
  sbc     xh, xh, r12
LOCAL_LABEL(add_swapped):
  // Keep the sign and exponent of the larger input, to use as the sign and
  // exponent of the output (up to carries and overflows). Also calculate the
  // exponent difference, which tells us how far we'll need to shift y's
  // mantissa right to add it to x's.
  //
  // The shifted-right values will include the sign bits as well as the
  // exponents, but both sign bits are the same, so they'll cancel.
  lsr     r4, xh, #20            // r4 = initial sign+exponent of the output
  sub     r12, r4, yh, lsr #20   // r12 = exponent difference

  // Clear the exponents and signs off the numbers to prepare for the addition.
  // (We reuse the value 0xffc00000 that we left in r14 on entry: ASRing that
  // right by 2 gives 0xfff00000, just the bit mask we wanted.)
  //
  // Also OR in the leading 1 bit of y's mantissa, so that when we shift it
  // right and add, it will be included in the addition.
  //
  // (It's cheaper not to bother doing the same for x, unless the addition
  // carries into the exponent.)
  bic     xh, xh, r14, asr #2
  bic     yh, yh, r14, asr #2
  orr     yh, yh, #1 << 20

LOCAL_LABEL(add_doadd):
  // Here we perform the actual addition. We either fell through from the code
  // above, or jumped back to here after handling an input denormal.
  //
  // We get here with:
  //   Operands known to be numeric rather than zero/infinity/NaN;
  //   xh:xl = mantissa of larger operand, with low bit at the bottom of xl
  //   yh:yl = mantissa of smaller operand, with low bit at the bottom of yl
  //   r4 = result sign and exponent (in low 12 bits);
  //   r12 = exponent difference.
  //
  // For normal inputs, the mantissa of y will have the leading bit set.
  // Denormals will leave that bit clear, treating the number as 0.[mantissa] x
  // 2^(fixed exponent) instead of renormalising to 1.[mantissa] x 2^(variable
  // exponent) as a multiplication would want.

  // The main addition involves shifting y right by the exponent difference in
  // r12, and adding it to x. This must be done differently depending on how
  // big the exponent difference is. Start by checking if it's at most 32.
  rsbs    r14, r12, #32
  blo     LOCAL_LABEL(add_bigshift)

  // The exponent difference is 32 or less. The test above also left
  // (32-difference) in r14, which is the amount we need to shift yh left by to
  // recover the bits that the right shift will lose off the bottom.
#if !__thumb__
  // Add the right-shifted parts of yh and yl to xh and xl, keeping the carry
  // in between if any.
  adds    xl, xl, yl, lsr r12
  adc     xh, xh, yh, lsr r12
  // Now add the remainder of yh to the low word, again checking for a carry.
  adds    xl, xl, yh, lsl r14
  adcs    xh, xh, #0
#else
  // Thumb can't fold a register-controlled shift into an add, so we must use
  // two separate instructions in each case.
  //
  // We don't have any more spare registers, so we'll use r14 as a temporary
  // register to hold each shifted value before adding it to something. This
  // clobbers the inverted shift count in r14, which we're going to need again
  // during rounding, so we must recompute it after the additions are complete.
  // (It would cost more cycles to avoid that awkwardness by pushing and
  // popping an extra register around the whole function.)
  //
  // To avoid recomputing r14 _twice_, we do the addition of (yh shifted left)
  // first, so we can use the value in r14 before clobbering it at all.
  lsl     r14, yh, r14
  adds    xl, xl, r14
  adcs    xh, xh, #0
  // Now do the addition of (yh shifted right) and (yl shifted right).
  lsr     r14, yl, r12
  adds    xl, xl, r14
  lsr     r14, yh, r12
  adc     xh, xh, r14
  // And now reconstruct the inverted shift count, for use later.
  rsb     r14, r12, #32
#endif

  // If that addition carried into bit 20 of xh, then the number has increased
  // its exponent. Diverge into a completely separate code path for that case,
  // because there we must check for overflow.
  cmp     xh, #1 << 20
  bhs     LOCAL_LABEL(add_carry)

  // Here, on the non-carrying path, we don't need to check for overflow at
  // all. If there is an overflow it can only be due to rounding up, so the
  // overflowed mantissa will be all zeroes, so the naively generated output
  // will look like the correct infinity anyway.
  //
  // Recombine the mantissa with the sign + exponent (in r4) via addition.
  add     xh, xh, r4, lsl #20
  // Now our number is complete apart from rounding.

LOCAL_LABEL(add_nocarry):
  // This is the general rounding path for additions that didn't carry into the
  // next exponent. We come here with the unrounded output in xh:xl, and yl and
  // r14 set up so that (yl << r14) consists of all the bits shifted off the
  // bottom of y's mantissa, or at least some approximation to them good enough
  // to make the right rounding decision.
  //
  // Perform that shift, which sets the N flag if we need to round.
  lsls    yl, yl, r14

  // We're done with our two extra registers, so we can pop them.
  pop     {r4, r14}

  // If N is clear, we're rounding down (or the result was exact), and we know
  // there was no overflow either, so xh:xl contains the correct output and we
  // can return immediately.
  bxpl    lr

  // Otherwise, we're rounding up, or rounding to even. Start by incrementing
  // the low word of the output.
  adds    xl, xl, #1

  // The obvious thing to do next would be to ADC xh, xh, #0, propagating any
  // carry from that ADDS, and completing the addition of 1 to the 64-bit value
  // in xh:xl. But we can do better, by doing a combined test for that carry
  // _and_ round-to-even, and returning as quickly as possible in the common
  // case where neither has happened.
  //
  // The Z flag is set if the addition to xl carried, and clear if it didn't.
  // So if Z is clear, we also test the bits of yl below the round bit. Then if
  // Z is still clear, there was no carry into xh _and_ no round to even, so we
  // can return.
  lslsne  yl, yl, #1
  bxne    lr

  // Now we know that we've just incremented xl, and either or both of these
  // things is true:
  //
  //  1. this is a halfway case that needs rounding to even
  //  2. the increment of xl wrapped it round from 0xFFFFFFFF to 0
  //
  // We can reliably tell if #2 is true by checking if xl = 0. If that is so,
  // we must increment xh. On the other hand, if xl != 0, then #1 must be true,
  // so we clear the low bit of xl to complete the round-to-even.
  //
  // What if _both_ are true? Luckily, it doesn't matter, because if xl = 0
  // then its low bit is already clear, so it makes no difference whether we
  // clear it or not.
  cmp     xl, #0                  // is xl 0?
  bicne   xl, xl, #1              // if not, then round to even
  adceq   xh, xh, #0              // if so, then increment xh
  bx      lr

LOCAL_LABEL(add_bigshift):
  // We come here from dadd_doadd if y's mantissa must be shifted right by more
  // than 32 bits. So all of yl is going to be shifted off the bottom, not
  // _even_ into the bit that determines rounding up or down. Therefore we can
  // approximate it well enough by a single bit at the bottom of yh, which is 1
  // if any bit of yl is 1.
  //
  // We put the modified value in yl, which is where the rounding code (shared
  // with the case for shift <= 32 bits) will expect to find the value it has
  // to shift left to make the round word.
  cmp     yl, #1                  // set C if yl >= 0
  adc     yl, yh, yh              // shift yh left 1, putting C at the bottom

  // Calculate shift counts. r12 is adjusted down by 32 so it tells us how much
  // to shift yh right by when adding; r14 is the distance to shift yl left by
  // to make the round word (again where the shared rounding code will expect
  // to find it).
  //
  // The second instruction also has the side effect of checking whether the
  // shift count in r12 is greater than 31, which we'll use in a moment.
  sub     r12, r12, #32
  rsbs    r14, r12, #31

  // Double precision exponents are bigger than 8 bits, so it's possible that
  // the exponent difference is > 255. AArch32 shift operations tolerate shifts
  // bigger than the size of the word, but only up to 255, because they only
  // look at the low 8 bits. So we must detect that r12 was huge, and handle it
  // specially.
  //
  // In this situation we reset r14 to 0, so that the rounding code will not
  // shift yl left at all. Since the top bit of yl is clear (we made yl by
  // shifting the top word of a mantissa left by 1, so its highest set bit is
  // at most bit 21), the effect is to consider _all_ of y's mantissa to be
  // lower than the round bit.
  movlo   r14, #0

  // Do the actual addition, again conditionalised on the result of checking
  // whether the shift count r12 was too big.
#if !__thumb__
  addshs  xl, xl, yh, lsr r12
#else
  lsrhs   yh, yh, r12
  addshs  xl, xl, yh
#endif

  // Recombine the (unrounded) output mantissa with the output sign and
  // exponent in r4. This also propagates any carry from xl into xh, from the
  // addition. (Luckily the condition for skipping the addition also implies
  // C=0, so in that situation, the ADC is still harmless.)
  adc     xh, xh, r4, lsl #20

  // Check whether the addition carried into the exponent field, by seeing if
  // the exponent that ended up at the top of xh is the same as the one in r4
  // that we just added. If it is the same (no carry) then we can go to
  // dadd_nocarry to do the easy version of rounding that doesn't also need to
  // check overflow.
  cmp     r4, xh, lsr #20
  beq     LOCAL_LABEL(add_nocarry)

  // Otherwise, the addition has carried into the exponent. Subtract the
  // exponent and sign off again, because dadd_carry (again shared with the
  // small-shift code) will need those not to be in xh, because it will need to
  // shift just the mantissa down by a bit.
  sub     xh, xh, r4, lsl #20

LOCAL_LABEL(add_carry):
  // We get here from both shift branches if magnitude addition overflowed the
  // input mantissa, so that the output will have an exponent one larger than
  // the larger input.
  //
  // xh:xl was the larger input mantissa _without_ its leading 1, which we then
  // added y's mantissa to. So before we shift down, we must put on the
  // explicit leading 1.
  add     xh, xh, #1 << 20
  lsrs    xh, xh, #1
  rrxs    xl, xl
  // Now we can put the sign and exponent back on.
  add     xh, xh, r4, lsl #20

  // The right shift left the round bit in C. So if that's clear, we're not
  // rounding up; we only have to check for overflow and then we can return.
  bcc     LOCAL_LABEL(add_check_overflow_pop)

  // Otherwise, set up for the combined dadd_roundeven_or_roundup_carry code:
  // round up by incrementing the low word of xl, leaving the carry bit set if
  // xh needs to be incremented too. If that addition _didn't_ carry, make the
  // round word in r14 that's zero if we need to round to even. Then Z is set
  // in either case, and otherwise, we only have overflow checking left to do.
  adds    xl, xl, #1              // set Z if there's a carry
  lslsne  r14, yl, r14            // else set Z if we need to round to even
  pop     {r4, r14}
  bne     LOCAL_LABEL(add_check_overflow)     // if Z not set for either reason, done

LOCAL_LABEL(add_roundeven_or_roundup_carry):
  // Just as in the dadd_nocarry case above, here we know that we've just
  // incremented xl, and we either need to propagate a carry into xh, or we
  // need to round to even, or both. See the comment there for explanation of
  // these three instructions.
  //
  // The difference in this case is that after we've done that, we also need to
  // check for overflow, where dadd_nocarry knew that wasn't necessary.
  cmp     xl, #0                  // is xl 0?
  bicne   xl, xl, #1              // if not, then round to even
  adceq   xh, xh, #0              // if so, then increment xh
  // We come here with a result ready to be returned, except that we have to
  // check it for overflow first.
LOCAL_LABEL(add_check_overflow):
  lsl     yh, xh, #1              // move exponent into top 11 bits of yh
  cmp     yh, #0xFFE00000         // if yh >= this, then exponent is all 1s
  bxlo    lr                      // otherwise, no overflow

  // If we haven't just returned, then we have an overflow. In addition we can
  // only overflow by up to a factor of 2, so the sign bit in xh is still
  // correct, and even the exponent has all its bits set. We only need to clear
  // the mantissa.
  mov     xl, #0                   // clear low word
  lsrs    xh, xh, #20
  lsls    xh, xh, #20
  bx      lr

  // Alternative entry point to dadd_check_overflow above, for use when the
  // registers pushed at the start of the function haven't been popped yet.
LOCAL_LABEL(add_check_overflow_pop):
  pop     {r4, r14}
  b       LOCAL_LABEL(add_check_overflow)

LOCAL_LABEL(add_uncommon):
  // We come here from the start of the function if we detected that either
  // input had exponent 0x7ff or 0x000: that is, at least one operand is a NaN,
  // infinity, denormal or zero.
  //
  // First detect whether there are any NaNs or infinities, by checking more
  // specifically if either input has exponent 0x7ff. We take advantage of
  // knowing that r14 was set to 0xFFC00000 in the function prologue, so we can
  // make a useful constant for this test by adjusting it.
  orr     r14, r14, #0x00200000   // now r14 = 0xFFE00000
  bics    r4, r14, xh, lsl #1     // if x has exponent 0x7ff, this sets r4=0
  bicsne  r4, r14, yh, lsl #1     // and similarly for y
  beq     LOCAL_LABEL(add_naninf)             // so if either set Z, we have a NaN or inf

  // Now we've ruled out NaNs and infinities. With NaNs gone, it's safe to flip
  // the signs of the inputs (which only mattered for returning the right NaN).
  // So check if the signs are the same, and cross-jump to dsub_zerodenorm
  // (magnitude subtraction involving a zero or denormal) if not. Meanwhile,
  // that will cross-jump back to here in the opposite case.
  teq     xh, yh
  eormi   yh, yh, #1 << 31
  bmi     LOCAL_LABEL(sub_zerodenorm)
LOCAL_LABEL(add_zerodenorm):
  // Now we know we're doing a magnitude addition, involving at least one zero
  // or denormal, and no NaNs or infinities.
  //
  // Sort the operands into magnitude order so that x >= y, exactly as in the
  // main code path.
  subs    r4, xl, yl          // compare inputs, also keeping x-y
  sbcs    r12, xh, yh
  bhs     LOCAL_LABEL(add_zerodenorm_swapped) // if x>=y then branch round the swap
  adds    yl, yl, r4          // otherwise turn y into x by adding (x-y)
  adc     yh, yh, r12
  subs    xl, xl, r4          // and turn x into y by subtracting it
  sbc     xh, xh, r12
LOCAL_LABEL(add_zerodenorm_swapped):
  // Set up the output sign+exponent, and the exponent difference, again
  // exactly as in the main code path.
  lsr     r4, xh, #20            // r4 = initial sign+exponent of the output
  sub     r12, r4, yh, lsr #20   // r12 = exponent difference

  // With the operands sorted so that y is smallest, and knowing there's at
  // least one zero or denormal present, we know furthermore that if there's
  // zero at all then it's y. And if y=0, then _whatever_ is in x is the right
  // answer to return from the whole operation, whether it's another zero, a
  // denormal, or normalised.
  orrs    r14, yl, yh, lsl #1     // test all bits of y except the sign bit
  popeq   {r4, pc}                // if they're all zero, we're done

  // Otherwise, there are no zeroes, so y must be denormal, and we don't yet
  // know if x is denormal too.
  //
  // If x isn't denormal, we rejoin the main code path for adding normalised
  // numbers, with everything set up as dadd_doadd expects. It's easiest to
  // represent the denormal y the same way the FP format does, as a mantissa
  // without its leading bit set, shifted by the same amount as normalised
  // numbers of the lowest exponent. (Renormalising via CLZ is more work, and
  // not needed for addition.)
  //
  // To tell the main code that y's mantissa should be shifted by the same
  // amount as a number with exponent 0x001, we must adjust the exponent
  // difference r12 by one, because we've already made that by subtracting the
  // _raw_ exponent values.

  lsls    r14, r4, #21          // output exp = 0? If so, x is denormal too
  bic     xh, xh, r4, lsl #20   // clear sign+exponent from top of x
  bicne   yh, yh, #1 << 31      // if x not denormal, clear sign of y
  subne   r12, r12, #1          //   and adjust exponent difference
  bne     LOCAL_LABEL(add_doadd)            //   and rejoin the main path

  // If we didn't take that branch, then both operands are denormal. In that
  // situation we can simply do a 64-bit _integer_ addition of the values we
  // have already! Both inputs represent numbers less than 2^52, with the same
  // exponent; so adding them produces a number less than 2^53, which means
  // it's either still a denormal, or if the addition carried into bit 52 then
  // it's become a normalised number, with the mantissa still scaled by the
  // same factor relative to the true value.
  //
  // The only tricky part is the sign bit. But we cleared that out of xh above,
  // and haven't cleared it out of yh, so there's exactly one copy of it
  // involved in this addition. So the sign bit will end up correct at the top
  // of xh too.
  adds    xl, xl, yl
  adc     xh, xh, yh
  pop     {r4, pc}

LOCAL_LABEL(add_naninf):
  // We come here knowing that at least one operand is either NaN or infinity.
  // If there's a NaN, we can tailcall __dnan2 to do the right thing. Pop our
  // stacked registers first: we won't need that much spare space any more, and
  // it makes the tailcall easier if we've already done it.
  pop     {r4, r14}

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

LOCAL_LABEL(add_inf):
  // Now we know there are no NaNs. Therefore there's at least one infinity. If
  // we have two infinities of opposite sign, that's an invalid operation and
  // we must return NaN; this happens if and only if x XOR y is all zero except
  // for the top bit.
  eor     r12, xh, yh
  cmp     r12, #0x80000000
  eorseq  r12, xl, yl
  beq     LOCAL_LABEL(addsub_return_nan)

  // Otherwise, only one sign of infinity is involved in our addition, so
  // return whichever operand is the infinity. Since we know there are no NaNs,
  // we can identify an infinity from just its exponent.
  lsl     r12, xh, #1
  cmp     r12, #0xFFE00000
  bxeq    lr
  movs    xh, yh
  movs    xl, yl
  bx      lr

LOCAL_LABEL(addsub_return_nan):
  // Return the default NaN, in the case of adding +inf to -inf.
  movw    xh, 0x7ff8
  lsls    xh, xh, #16        // 0x7ff80000 is the high word of the default NaN
  mov     xl, #0             // and the low word is 0
  bx      lr

END_COMPILERRT_FUNCTION(__aeabi_dadd)

DEFINE_COMPILERRT_FUNCTION(__aeabi_drsub)
  // Reversed subtraction, that is, compute y-x, where x is in r0/r1 and y in
  // r2/r3.
  //
  // We could implement this by simply swapping the register pairs. But the
  // point of having a reversed-subtract in the first place is to avoid the
  // caller having to do that, so if we do it ourselves, it wastes all the time
  // they saved. So instead, on the fast path, we redo the sign check our own
  // way and branch to dadd_magnitude or dsub_magnitude.

  push    {r4, r14}

  // Start by testing for uncommon operands in the same way as dadd.
  ldr     r14, =0xFFC00000
  add     r12, xh, #1 << 20   // r12 has the adjusted version of x's exponent
  add     r4, yh, #1 << 20    // and r4 the adjusted version of y's
  tst     r14, r12, lsl #1    // test the top 10 exponent bits of each
  tstne   r14, r4, lsl #1
  beq     LOCAL_LABEL(rsub_uncommon)      // and branch out of line if either is 0

  // Check if the signs are equal, and branch to one or the other of
  // dadd_magnitude and dsub_magnitude.
  //
  // If the signs are unequal, then y-x is a magnitude addition: we negate x so
  // that we're computing y + (-x), in which both values have the same sign and
  // go to dadd_magnitude. If the signs are equal then y-x is a magnitude
  // subtraction, equal to (-x) - (-y), so we negate both operands and go to
  // dsub_magnitude. Since x needs to be negated in both cases, we can do that
  // unconditionally.
  teq     xh, yh              // N set for a magnitude addition
  eor     xh, xh, #1 << 31    // negate x unconditionally
  bmi     LOCAL_LABEL(add_magnitude)      // branch away for magnitude addition
  eor     yh, yh, #1 << 31    // otherwise, negate y too
  b       LOCAL_LABEL(sub_magnitude)      // and do a magnitude subtraction

LOCAL_LABEL(rsub_uncommon):
  // Any uncommon operands to drsub are handled by just swapping the two
  // operands and going to dsub's handler. We're off the main fast path now, so
  // there's no need to try to optimise it any harder.
  eor     xh, xh, yh
  eor     xl, xl, yl
  eor     yh, yh, xh
  eor     yl, yl, xl
  eor     xh, xh, yh
  eor     xl, xl, yl
  b       LOCAL_LABEL(sub_uncommon)

END_COMPILERRT_FUNCTION(__aeabi_drsub)

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__subdf3)
  push {r4, lr}
  VMOV_FROM_DOUBLE(r0, r1, d0)
  VMOV_FROM_DOUBLE(r2, r3, d1)
  bl __aeabi_dsub
  VMOV_TO_DOUBLE(d0, r0, r1)
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__subdf3, __aeabi_dsub)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_dsub)
  // Main entry point for subtraction.

  push    {r4, r14}

  // Start by testing for uncommon operands in the same way as dadd.
  ldr     r14, =0xFFC00000
  add     r12, xh, #1 << 20   // r12 has the adjusted version of x's exponent
  add     r4, yh, #1 << 20    // and r4 the adjusted version of y's
  tst     r14, r12, lsl #1    // test the top 10 exponent bits of each
  tstne   r14, r4, lsl #1
  beq     LOCAL_LABEL(sub_uncommon)       // and branch out of line if either is 0

  // Check the signs, and if they're unequal, cross-jump into dadd to do
  // magnitude addition. (Now we've excluded NaNs, it's safe to flip the sign
  // of y.)
  teq     xh, yh
  eormi   yh, yh, #1 << 31
  bmi     LOCAL_LABEL(add_magnitude)
LOCAL_LABEL(sub_magnitude):
  // If we get here, we're subtracting operands with equal signs (i.e. a
  // magnitude subtraction). First thing to do is put operands in magnitude
  // order, so that x >= y. However, if they are swapped, we must also negate
  // both of them, since A - B = (-B) - (-A). We do this by flipping the top
  // bit of the value we add/subtract to each input to perform the swap
  subs    r4, xl, yl          // compare inputs, also keeping x-y
  sbcs    r12, xh, yh
  bhs     LOCAL_LABEL(sub_swapped)        // if x>=y then branch round the swap
  eor     r12, r12, #1 << 31  // flip the top bit of x-y
  adds    yl, yl, r4          // so that this addition turns y into x+TOPBIT
  adc     yh, yh, r12
  subs    xl, xl, r4          // and this subtraction turns x into y-TOPBIT
  sbc     xh, xh, r12
LOCAL_LABEL(sub_swapped):
  // Keep the sign and exponent of the larger input, to use as the sign and
  // exponent of the output (up to carries and overflows). Also calculate the
  // exponent difference, which tells us how far we'll need to shift y's
  // mantissa right to add it to x's.
  //
  // As in dadd, the values being subtracted both include the sign bit, but
  // we've already ensured the sign bits are the same (if we came here from
  // dadd then we flipped the sign of y), so as in dadd, they cancel.
  lsr     r4, xh, #20
  sub     r12, r4, yh, lsr #20

  // Isolate the two mantissas.
  bic     xh, xh, r4, lsl #20
  bic     yh, yh, r14, asr #2     // 0xffc00000 ASR 2 = 0xfff00000

  // Negate the mantissa of y, so that we can compute the difference using
  // ADD/ADC. As a side effect we also add in the leading bit of y's mantissa,
  // by subtracting y from 0xfff0000000000000 instead of from 0.
  rsbs    yl, yl, #0
#if !__thumb__
  rsc     yh, yh, r14, asr #2     // 0xffc00000 ASR 2 = 0xfff00000
#else
  // Thumb has no RSC, so simulate it by bitwise inversion and then ADC
  mvn     yh, yh
  adc     yh, yh, r14, asr #2     // 0xffc00000 ASR 2 = 0xfff00000
#endif

LOCAL_LABEL(sub_dosub):
  // Here we perform the actual subtraction. We either fell through from the
  // code above, or jumped back to here after handling an input denormal.
  //
  // We get here with:
  //   Operands known to be numeric rather than zero/infinity/NaN;
  //   xh:xl = mantissa of larger operand, with low bit at the bottom of xl
  //   yh:yl = negated mantissa of smaller operand, similarly
  //   r4 = result sign and exponent (in low 12 bits);
  //   r12 = exponent difference.
  //
  // For normal inputs, the value in yh:yl will be as if the mantissa of y had
  // the leading bit set before negating it. For denormal y, the mantissa will
  // have been negated without setting that bit, similarly to dadd.

  // As in dadd, we start by separating off the case where we're shifting the
  // mantissa of y right by more than 32 bits.
  rsbs    r14, r12, #32
  blo     LOCAL_LABEL(sub_bigshift)

  // The exponent difference is 32 or less. The test above also left
  // (32-difference) in r14, which is the amount we need to shift yh left by to
  // recover the bits that the right shift will lose off the bottom.
#if !__thumb__
  // Add the right-shifted parts of yh and yl to xh and xl, keeping the carry
  // in between if any.
  adds    xl, xl, yl, lsr r12
  adc     xh, xh, yh, asr r12
  // Now add the remainder of yh to the low word, again checking for a carry.
  adds    xl, xl, yh, lsl r14
  adcs    xh, xh, #0
#else
  // The Thumb version of the addition, which must do each register-controlled
  // shift in a separate instruction from the addition. This works the same as
  // the dadd version, except that we use ASR to shift yh right, because yh:yl
  // contains a negative signed integer.

  // As in dadd, start by adding (yh shifted left), so as not to waste the
  // value we've already set up in r14.
  lsl     r14, yh, r14
  adds    xl, xl, r14
  adcs    xh, xh, #0
  // Then add (yh shifted right) and (yl shifted right).
  lsr     r14, yl, r12
  adds    xl, xl, r14
  asr     r14, yh, r12
  adcs    xh, xh, r14
  // And now reconstruct the inverted shift count, for use later.
  rsb     r14, r12, #32
#endif

  // We know we had x >= y before the subtraction. So x-y is still a number of
  // the same sign, but its exponent might have reduced. If we'd set the
  // leading bit on x's mantissa before subtracting, we'd be able to tell this
  // by testing if it was still set. But in fact we didn't, so the question is
  // whether x's mantissa without the leading bit is still even positive.
  //
  // The last ADCS (in either of the Arm and Thumb code sequences above) will
  // have set the N flag if x < 0, which is the case where the exponent has
  // reduced. Branch out of line for that case.
  bmi     LOCAL_LABEL(sub_borrow)

LOCAL_LABEL(sub_noborrow):
  // This is the easy case: the exponent of x has stayed the same, so there's
  // no possibility of underflow. All we have to do is put the pieces of the
  // result back together, round, and return.

  // Recombine x's mantissa with the output sign and exponent.
  add     xh, xh, r4, lsl #20

  // Make the word of bits shifted off the bottom of y's mantissa, with the
  // topmost bit indicating whether we round up or down, and the rest used to
  // determine whether to round to even.
  lsls    yl, yl, r14

  // If the top bit of the round word is clear, then we're rounding down, so
  // the value in xh:xl is already correct and we can return.
  poppl   {r4, pc}

  // Otherwise, start by rounding up. As in dadd, we make the Z flag do double
  // duty: it's initially set by the ADDS to indicate a carry into the high
  // word, and then if that doesn't happen then we have another chance to set
  // it if the round word indicates an exact halfway case. So we can return
  // early in the common case where neither of those things happened.
  adds    xl, xl, #1
  cmpne   yl, #0x80000000
  popne   {r4, pc}

  // Now if xl=0 then we must increment xh (the addition from rounding carried
  // into the high word). Otherwise we must round to even, by clearing the low
  // bit of xl. As in dadd, it's possible that _both_ conditions are true at
  // once, but in that situation, the fact that xl=0 means if makes no
  // difference whether we clear its low bit or not.
  cmp     xl, #0              // do we need to increment xh?
  addeq   xh, xh, #1          // if so, do it
  bicne   xl, xl, #1          // otherwise, round to even
  pop     {r4, pc}

LOCAL_LABEL(sub_bigshift):
  // We come here from dsub_dosub if y's mantissa must be shifted right by more
  // than 32 bits.
  //
  // In dadd_bigshift we concluded that all of yl could be condensed into a
  // single bit at the bottom of the round word, because it could only affect
  // round-to-even. However, in subtraction, that's not true, because we might
  // renormalise: if the input exponents differ by exactly 33, and the
  // subtraction reduces the exponent by 1, then the top bit of yl might become
  // the round bit again. So we must make our round word by shifting two extra
  // bits on to the bottom of yh: first the topmost bit of yl, then a single
  // bit indicating whether any of the rest is nonzero.
  //
  // As in dadd_bigshift, we make this new round word in yl, leaving yh
  // unmodified so that we can use it for the actual shift-and-add.
  //
  // (For these purposes, we only have to worry about renormalisation by _one_
  // bit. If the output exponent reduces by 2 or more, it must be because the
  // input exponents were so close that the output is exact anyway, so a round
  // word isn't needed at all.)
  adds    r14, yl, yl         // put the top bit of yl into C
  adc     yl, yh, yh          // and shift it in to the bottom of yh
  cmp     r14, #1             // set C if anything below that bit was nonzero
  adc     yl, yl, yl          // shift that in to yl as well

  // Calculate shift counts. r12 is how far to shift yh right when adding; r14
  // is how far to shift yl left to make the round word (subtracted from 30
  // instead of 32 to account for the two bits we just shifted in at the bottom
  // of yl).
  //
  // If the latter shift count goes negative, then we can't use it. Branch to
  // another handler for _really_ big exponent differences.
  sub     r12, r12, #32
  rsbs    r14, r12, #30
  blo     LOCAL_LABEL(sub_hugeshift)

  // Shift yh right and add it to x, to produce the unrounded output mantissa.
#if !__thumb__
  adds    xl, xl, yh, asr r12
#else
  // In Thumb we must do the register-controlled shift and addition separately
  asr     r12, yh, r12
  adds    xl, xl, r12
#endif
  // The top half of the addition, propagating a carry from xl into xh. Since
  // yh was a negative number and we arithmetically shifted it right, the value
  // we add to xh is 0xFFFFFFFF rather than 0, as if we'd sign-extended that
  // negative number to 64 bits.
  adcs    xh, xh, #-1

  // As in the small-shift case above, if this has left a positive value in
  // xh:xl, it means the exponent hasn't changed, so we can go to the easy
  // epilogue code in dsub_noborrow.
  bpl     LOCAL_LABEL(sub_noborrow)

LOCAL_LABEL(sub_borrow):
  // We come here from either of the small-shift or large-shift versions of the
  // subtraction step, if the subtraction caused xh:xl to go negative. This
  // means that the result of the subtraction is less than the smallest
  // possible value with x's exponent. In other words, the output will have a
  // smaller exponent, and we must shift the mantissa left and put some bits
  // back in from yl (which contains the bits of y shifted off the bottom).
  //
  // The most important question in this situation is: do we have to shift the
  // mantissa left by only one bit, or by more than one? It's important because
  // in the case where we shift left by more than one bit, no rounding can
  // possibly be needed: if x >= 2^k but x-y < 2^{k-1}, then y > 2^{k-1}, so
  // the exponents of x and y differ by at most 1. Therefore the lowest set bit
  // in the true difference x-y (before rounding) can't possibly be any lower
  // than the bit just off the bottom of x's mantissa, and we're shifting left
  // by at least 1, so that will be part of the output mantissa. So in this
  // case the result must be exact.
  //
  // (This is not normally considered a good thing from the point of view of
  // the user! Subtracting two very close values and getting a result that has
  // a lot of mantissa bits zero at the bottom is called 'significance loss'
  // and can be a cause of numerical instability. But whether the client code
  // _likes_ it or not, the IEEE standard is very clear that we must return the
  // value with lots of trailing 0 bits, which can't need any rounding.)
  //
  // On the other hand, if we shift left by only one bit, then the value we
  // subtracted from x could have been almost arbitrarily small, so there's
  // lots of scope for bits of y to have been shifted off the bottom to cause
  // rounding.
  //
  // Conclusion: we either shift left 1 and have to figure out rounding, or we
  // shift left more than 1 and have to figure out the right shift count, but
  // never both.

  // On entry to here, (yl << r14) gives the bits shifted off the bottom of
  // xh:xl. Shift xh:xl up by one, bringing the high bit of that back in.
  //
  // If we're shifting left by only one bit, then the mantissa is now at its
  // correct position and yl is the round word. On the other hand, if we're
  // shifting by more, then all the output mantissa bits we need are now in
  // xh:xl, and there aren't any in yl that still need to be salvaged.
  add     r14, r14, #1            // we want to shift yl one extra bit left
  lsls    r14, yl, r14            // do the shift, leaving the top bit in C
  adcs    xl, xl, xl              // shift that in to the bottom of xl
  adc     xh, xh, xh              // and propagate into xh

  // Our next task is to find out which case we're in: shift by one bit and
  // round, or figure out how many more bits to shift by? We can determine this
  // by looking at bit 20 of xh: if that's 0 then we need to shift further.
  //
  // But to save instructions, we fold that test together with a test for
  // another awkward case: was the input exponent in r4 equal to 1? If so, then
  // it's been decremented to 0, which means the result of the subtraction is a
  // denormal. (Separately from that, we might _also_ get a denormal if
  // significance loss has occurred, even if the exponent in r4 was larger.)
  //
  // To do both of these tests at once, we add the original output exponent in
  // r4 back in to xh, _shifted left by an extra bit_, as if we'd added it
  // before doing the shift above. This loses the sign bit off the top, and
  // since the top 11 bits of xh are all 1, has the same result as decrementing
  // r4. So bit 20 of xh is unaffected (it's still 0 if we need to shift
  // further), and bits 21 and upwards are all zero if the output might be
  // denormal.
  //
  // The Arm condition code LS (unsigned lower-or-same) is implemented by
  // testing if C=0 or Z=1. That's just what we need! Having made our modified
  // version of xh, shift it right so that bit 20 goes off the bottom into the
  // carry flag. Then C=0 means bit 20 of xh was clear and we need to shift
  // further; Z=1 means the exponent has decremented from 1 to 0 and we're
  // returning a denormal; if _either_ is true, then the BLS will send us out
  // of line.

  add     r12, xh, r4, lsl #21    // make test value (keeping the original xh)
  lsrs    r12, r12, #21           // set C and Z to the values we want to test
  bls     LOCAL_LABEL(sub_renorm_or_denorm)   // branch out of line if C=0 or Z=1

  // If we haven't taken that branch, then we now have our mantissa in the
  // correct position _and_ we're confident that the output is a normalised
  // number. So we only have rounding left to do.
  //
  // Put the sign and exponent back on the output. Because the bits in xh's
  // exponent field are still all 1s, this decrements the exponent in r4 by
  // one, which is just what we want.
  add     xh, xh, r4, lsl #20

  // The round bit is at the top of r14, so we can add it to the bottom of xl
  // by a right shift.
  //
  // If this addition carries off the top of xl, then C and Z will both be set.
  // If C is not set, then Z might still be set because xl was already zero.
  adds    xl, xl, r14, lsr #31
  // We only need to check for round-to-even if there wasn't a carry, because
  // if there was a carry, xl = 0 and so clearing its low bit won't make a
  // difference anyway. So in the C=0 case, we now clobber the potentially
  // misleading value left in Z by the previous instruction, and replace it
  // with the result of checking r14 against the exact halfway value of the
  // round word.
  cmpcc   r14, #0x80000000
  // Now if Z is clear, we don't have to round to even _or_ propagate a carry
  // into xh, so we're done.
  popne   {r4, pc}

  // Otherwise, we have to either round to even, or increment xh. We increment
  // xh exactly if xl = 0, because the case where xl=0 without rounding up
  // would have taken the early return: the ADDS would have left C clear, so
  // the CMPCC would have checked r14 against 0x80000000, and would have
  // compared unequal because the top bit of r14 would have been claer.
  cmp     xl, #0                  // is xl zero?
  addeq   xh, xh, #1              // if so, increment xh to propagate carry
  bicne   xl, xl, #1              // otherwise, clear xl bit 0 to round to even
  pop     {r4, pc}

LOCAL_LABEL(sub_renorm_or_denorm):
  // We come here from the tricky combined test above, where we set C=0 if the
  // output mantissa still doesn't have its leading bit set, and Z=1 if the
  // exponent has already decreased to 0 so that the output will be denormal.
  //
  // In the latter case, we don't want to shift the mantissa any further up,
  // because we'd only have to shift it back down again. So branch again to
  // deal with that, or fall through to multiple-bit renormalisation.
  beq     LOCAL_LABEL(sub_already_denormal)

  // We'll want to adjust the exponent by the amount we shift. So split up the
  // sign and exponent, so that we can do arithmetic on the exponent without
  // the sign getting in the way.
  lsr     r12, r4, #11            // sign is now in r12 bit 0
  bic     r4, r4, #1 << 11        // exponent is in r4 all by itself

  // Add the leading bit of x's mantissa back in (at bit 21 rather than 20
  // because we already shifted left by one), to recover the full output
  // mantissa.
  //
  // As a side effect, this sets Z to indicate that the top word xh is all
  // zero, so now we know which of xh and xl we need to CLZ. It's easier to
  // separate the two cases than to try to deal with them in a combined code
  // path. We branch out of line for the xh=0 case, on the theory that the
  // larger the renormalization, the less likely it is, so the common case
  // stays in line.
  adds    xh, xh, #1 << 21
  beq     LOCAL_LABEL(sub_renorm_clz_xl)

  // There's a set bit somewhere in xh. Find it, and shift it up to bit 20.
  clz     yl, xh                  // distance from leading bit to bit 31
  subs    yl, yl, #11             // distance to bit 20, where we want it
  rsbs    yh, yl, #32             // work out the associated right shift
  lsls    xh, xh, yl              // shift xh upwards
#if !__thumb__
  orr     xh, xh, xl, lsr yh      // combine with the high bits of xl
#else
  // As usual, in Thumb we must do the register-controlled right shift and the
  // ORR separately.
  lsrs    yh, xl, yh
  orrs    xh, xh, yh
#endif
  lsls    xl, xl, yl              // finally, shift xl left

  // Adjust the exponent downward, to match the distance we just shifted the
  // mantissa upward.
  //
  // We adjust downward by an extra 2: one because we already shifted xh left
  // by one bit, and another because the leading bit of the renormalized
  // mantissa will increment it again.
  subs    r4, r4, yl
  subs    r4, r4, #2

LOCAL_LABEL(sub_renormed):
  // Here the two renormalization branches reconverge. The output mantissa in
  // xh:xl has been shifted up to the correct position, with its leading bit
  // present and in bit 20 of xh. r4 is the adjusted exponent, and the low bit
  // of r12 is the output sign.
  //
  // Recombine all the pieces. Since no rounding is needed on this path, the
  // output is correct and ready to return unless the exponent is too small.
  // The smallest valid exponent is 0, because it will be adjusted upwards by 1
  // by the leading mantissa bit. Since the last thing both branches did before
  // coming here was to update r4 using a flag-setting instruction, we can
  // therefore detect underflow by the N flag.
  add     xh, xh, r12, lsl #31
  add     xh, xh, r4, lsl #20
  poppl   {r4, pc}

  // Renormalisation made the exponent negative. We're well off the fast path
  // by now, so the simplest way to sort this out is to use the helper routine
  // __dunder.
  add     xh, xh, #3 << 29        // rebias exponent as __dunder will expect
  mov     r2, #0                  // rounding direction = 0 for an exact answer
  pop     {r4, lr}
  b       SYMBOL_NAME(__compiler_rt_dunder)

LOCAL_LABEL(sub_renorm_clz_xl):
  // This is the alternative renormalization code for the case where xh=0, so
  // that the highest remaining set bit in the mantissa is somewhere in xl.
  // Again we want to shift that all the way up to bit 20 of xh. The easiest
  // way is to shift it to the top of xl, and then shift that in turn by a
  // fixed distance to split it across xh[20..0] and xl[31..21], saving a
  // conditional decision about whether to shift up or down.
  //
  // However, there's another special case: on this branch, we might find out
  // that we've subtracted two _exactly_ equal values, not just nearly equal,
  // so the result is zero! To handle this quickly, we put the shifted-up
  // version of xl into xh instead of shifting it in place. Then, if it's zero,
  // we've just filled xh _and_ xl with zero bits, so we can return
  // immediately. (Since this function always uses round-to-nearest mode, an
  // output zero from subtracting like-signed inputs is unconditionally +0.)
  clz     yh, xl
  lsls    xh, xl, yh              // now xl has leading bit in bit 31
  popeq   {r4, pc}                // and if the answer is 0, just return it
  lsls    xl, xh, #21             // now set xl to the low bits of the mantissa
  lsrs    xh, xh, #11             // and xh to the high bits

  // Adjust the exponent down by the amount we shifted up, which is the CLZ
  // output (in yh), plus another 21 bits to get from the top bit of xl to bit
  // 20 of xh, plus 1 bit for the shift already performed before we did the
  // CLZ, plus 1 which the leading mantissa bit will undo when we add it to the
  // exponent. Then go back to dsub_renormed for the shared epilogue code.
  subs    r4, r4, yh
  subs    r4, r4, #23
  b       LOCAL_LABEL(sub_renormed)

LOCAL_LABEL(sub_hugeshift):
  // We came here in the case where the whole of y's mantissa was shifted down
  // so far that dsub_bigshift couldn't cope with it. In this situation the
  // result of the subtraction differs from the input x by under half a ULP, so
  // we just return the original x, which we recover by putting the sign and
  // exponent in r4 back together with the mantissa.
  add     xh, xh, r4, lsl #20
  pop     {r4, pc}

LOCAL_LABEL(sub_already_denormal):
  // We come here if the initial renormalization by one bit reduced the
  // exponent of x from 1 to 0, so that the output is denormal. In this
  // situation we don't need to call __dunder to figure out how far to shift
  // the result, because the answer is a constant: the mantissa was already in
  // the right place _before_ our one-bit left shift (denormals have the same
  // mantissa shift as normalised numbers with the smallest exponent), so all
  // we have to do is undo that left shift, and put the sign bit back on.
  movs    xh, xh, asr #1
  rrx     xl, xl
  add     xh, xh, r4, lsl #20

LOCAL_LABEL(sub_check_zero):
  // Here we have a denormal result in xh:xl, with its sign bit already in
  // place ... except that the mantissa might be all zeroes, in which case we
  // must clear the sign bit so as to return +0.
  pop     {r4, r14}
  orrs    r12, xl, xh, lsl #1     // EQ if all non-sign bits of x are zero
  bxne    lr                      // if that's not true, return our denormal
  movs    xh, #0                  // otherwise, clear xh completely
  bx      lr

LOCAL_LABEL(sub_uncommon):
  // We come here from the start of the function if we detected that either
  // input had exponent 0x7ff or 0x000: that is, at least one operand is a NaN,
  // infinity, denormal or zero.
  //
  // First detect whether there are any NaNs or infinities, by checking more
  // specifically if either input has exponent 0x7ff. We take advantage of
  // knowing that r14 was set to 0xFFC00000 in the function prologue, so we can
  // make a useful constant for this test by adjusting it.
  orr     r14, r14, #0x00200000   // now r14 = 0xFFE00000
  bics    r4, r14, xh, lsl #1     // if x has exponent 0x7ff, this sets r4=0
  bicsne  r4, r14, yh, lsl #1     // and similarly for y
  beq     LOCAL_LABEL(sub_naninf)             // so if either set Z, we have a NaN or inf

  // Now we've ruled out NaNs and infinities. With NaNs gone, it's safe to flip
  // the signs of the inputs (which only mattered for returning the right NaN).
  // So check if the signs are the same, and cross-jump to dadd_zerodenorm
  // (magnitude subtraction involving a zero or denormal) if not. Meanwhile,
  // that will cross-jump back to here in the opposite case.
  teq     xh, yh
  eormi   yh, yh, #1 << 31
  bmi     LOCAL_LABEL(add_zerodenorm)
LOCAL_LABEL(sub_zerodenorm):
  // Now we know we're doing a magnitude addition, involving at least one zero
  // or denormal, and no NaNs or infinities.
  //
  // Sort the operands into magnitude order so that x >= y, exactly as in the
  // main code path, including the EOR that negates both operands in the course
  // of swapping them.
  subs    r4, xl, yl          // compare inputs, also keeping x-y
  sbcs    r12, xh, yh
  bhs     LOCAL_LABEL(sub_zerodenorm_swapped) // if x>=y then branch round the swap
  eor     r12, r12, #1 << 31  // flip the top bit of x-y
  adds    yl, yl, r4          // so that this addition turns y into x+TOPBIT
  adc     yh, yh, r12
  subs    xl, xl, r4          // and this subtraction turns x into y-TOPBIT
  sbc     xh, xh, r12
LOCAL_LABEL(sub_zerodenorm_swapped):
  // Set up the output sign+exponent, and the exponent difference, again
  // exactly as in the main code path.
  lsr     r4, xh, #20
  sub     r12, r4, yh, lsr #20

  // With the operands sorted so that y is smallest, and knowing there's at
  // least one zero or denormal present, we know furthermore that if there's
  // zero at all then it's y. And if y=0, then we just return x, except that if
  // x=0 too we must fix up the sign of zero.
  orrs    r14, yl, yh, lsl #1     // test all bits of y except the sign bit
  beq     LOCAL_LABEL(sub_check_zero)         // if they're all zero, return x

  // Otherwise, there are no zeroes, so y must be denormal, and we don't yet
  // know if x is denormal too.
  //
  // If x isn't denormal, we rejoin the main code path for adding normalised
  // numbers, with everything set up as dadd_doadd expects. It's easiest to
  // represent the denormal y the same way the FP format does, as a mantissa
  // without its leading bit set, shifted by the same amount as normalised
  // numbers of the lowest exponent. (Renormalising via CLZ is more work, and
  // not needed for addition.)
  //
  // To tell the main code that y's mantissa should be shifted by the same
  // amount as a number with exponent 0x001, we must adjust the exponent
  // difference r12 by one, because we've already made that by subtracting the
  // _raw_ exponent values.
  lsls    r14, r4, #21          // output exp = 0? If so, x is denormal too
  bic     xh, xh, r4, lsl #20   // clear sign+exponent from top of x
  beq     LOCAL_LABEL(sub_both_denorm)      // if both inputs denormal, go elsewhere
  bic     yh, yh, #1 << 31      // if x not denormal, clear sign of y
  sub     r12, r12, #1          //   and adjust exponent difference
  // Now negate the mantissa of y and then rejoin the main path.
  rsbs    yl, yl, #0
#if !__thumb__
  rsc     yh, yh, #0
#else
  // Thumb has no RSC, so simulate it by bitwise inversion and then ADC
  mvn     yh, yh
  adc     yh, yh, #0
#endif
  b       LOCAL_LABEL(sub_dosub)

LOCAL_LABEL(sub_both_denorm):
  // If both inputs are denormal, then we can just subtract the mantissas like
  // ordinary integers. We've cleared the sign bit from x, but not from y, so
  // we'll get exactly one copy of the sign bit in the result. (Negating it
  // makes no difference!)
  subs    xl, xl, yl
  sbc     xh, xh, yh
  // Now go to dsub_check_zero, which will check if the answer is exactly zero,
  // and fix the sign bit if it is.
  b       LOCAL_LABEL(sub_check_zero)

  // Handle NaNs and infinities in subtraction.
LOCAL_LABEL(sub_naninf):
  // Look for NaNs and hand them off to __dnan2, exactly as in dadd_naninf.
  pop     {r4, r14}
  cmp     xl, #1
  adc     r12, xh, xh
  cmp     r12, #0xFFE00000
  bhi     SYMBOL_NAME(__compiler_rt_dnan2)
  cmp     yl, #1
  adc     r12, yh, yh
  cmp     r12, #0xFFE00000
  bhi     SYMBOL_NAME(__compiler_rt_dnan2)

  // Now we know there aren't any NaNs, we can deal with subtractions involving
  // an infinity by flipping the sign of y and letting dadd_inf deal with it.
  eor     yh, yh, #0x80000000
  b       LOCAL_LABEL(add_inf)

END_COMPILERRT_FUNCTION(__aeabi_dsub)

NO_EXEC_STACK_DIRECTIVE

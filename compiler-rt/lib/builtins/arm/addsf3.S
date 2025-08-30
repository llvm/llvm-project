//===-- addsf3.S - Adds two single precision floating point numbers--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __addsf3 (single precision floating point number
// addition with the IEEE-754 default rounding (to nearest, ties to even)
// function for the ARM Thumb1 ISA.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"
#define significandBits 23
#define typeWidth 32

	.syntax unified
	.text
  .thumb
  .p2align 2

DEFINE_AEABI_FUNCTION_ALIAS(__aeabi_fadd, __addsf3)

DEFINE_COMPILERRT_THUMB_FUNCTION(__addsf3)
  PUSH {r4,r5,r6,lr}

  MOVS    r5, #1
  LSLS    r5, r5, #31  // all cross-branches will expect to have r5==0x80000000

  // Extract the exponents into r2 and r3. In the process, test for all
  // uncommon values (infinities, NaNs, denormals and zeroes) and branch out of
  // line if any are found.
  //
  // Uncommon operands with exponent 0xFF (NaNs and infinities) "win" over
  // those with exponent 0 (zeroes and denormals), in the sense that if there's
  // one of each, the 0xFF one determines the result. But we check for exponent
  // 0 first, because that way we get it as a by-product of extracting the
  // exponents in the first place without needing a separate compare
  // instruction. So the zero/denorm handler will have to finish up the NaN
  // check as its first task.
  LSLS    r2, r0, #1
  LSLS    r3, r1, #1
  LSRS    r2, r2, #24
  BEQ     LOCAL_LABEL(fadd_zerodenorm_x)
  LSRS    r3, r3, #24
  BEQ     LOCAL_LABEL(fadd_zerodenorm_y)
  CMP     r2, #255
  BEQ     LOCAL_LABEL(fadd_naninf)
  CMP     r3, #255
  BEQ     LOCAL_LABEL(fadd_naninf)

  // Now we have two normalised numbers. If their signs are opposite, we should
  // be subtracting their magnitudes rather than adding, so cross-jump to fsub
  // (via a trampoline that negates y).
  MOVS    r4, r0
  EORS    r4, r4, r1         // set N if signs are unequal
  BMI     LOCAL_LABEL(fadd_sub)
LOCAL_LABEL(fadd_magnitude):
  // If we get here, we're adding operands with equal signs (i.e. a magnitude
  // addition). First thing to do is put the operands in magnitude order, so
  // that x >= y.
  SUBS    r4, r0, r1
  BHS     LOCAL_LABEL(fadd_swapped)
  SUBS    r0, r0, r4
  ADDS    r1, r1, r4
  // We must also swap the pre-extracted exponents here.
  EORS    r2, r2, r3
  EORS    r3, r3, r2
  EORS    r2, r2, r3
LOCAL_LABEL(fadd_swapped):
  // Keep the sign and exponent of the larger input, to use as the sign and
  // exponent of the output (up to carries and overflows). Also calculate the
  // exponent difference, which tells us how far we'll need to shift y's
  // mantissa right to add it to x's.
  LSRS    r6, r0, #23
  SUBS    r3, r2, r3

  // Extract both mantissas, moved up to the top of the word, with the leading
  // 1 made explicit. We put y's extracted mantissa in a different register
  // (r4), because we'll want to keep the original y for use in fadd_check_rte.
  LSLS    r0, r0, #8
  LSLS    r4, r1, #8
  ORRS    r0, r0, r5
  ORRS    r4, r4, r5

LOCAL_LABEL(fadd_doadd):
  // Here we perform the actual addition. We either fell through from the code
  // above, or jumped back to here after handling an input denormal.
  //
  // We get here with:
  //   Operands known to be numeric rather than zero/infinity/NaN;
  //   r0 = mantissa of larger operand (in high 24 bits);
  //   r4 = mantissa of smaller operand (in high 24 bits);
  //   r1 = original (or nearly so) smaller operand;
  //   r6 = result sign and exponent (in low 9 bits);
  //   r2 = exponent of x
  //   r3 = exponent difference.
  //
  // For normal inputs, the mantissa registers (r0,r4) will have the top bit
  // set. Denormals will leave that bit clear, treating the number as
  // 0.[mantissa] x 2^(fixed exponent) instead of renormalising to 1.[mantissa]
  // x 2^(variable exponent) as a multiplication would want.

  // Actually shift the smaller mantissa downwards and add them together.
  LSRS    r4, r4, r3
  ADDS    r5, r0, r4

  // If that addition carried off the top of r5, then the number has increased
  // its exponent. Diverge into a completely separate code path for that case,
  // because there we must check for overflow. We'll return to the label below
  // if no overflow.
  BCS     LOCAL_LABEL(fadd_carry)
LOCAL_LABEL(fadd_renormed):
  // Now we have the output mantissa in r5, with the leading bit at position
  // 31. The precise sum may be slightly more than that, if r4 != (y << r3).
  //
  // Shift the mantissa down to its final position, and use the carry flag (bit
  // shifted off the bottom) to see if we need to round.
  LSRS    r0, r5, #8
  BCC     LOCAL_LABEL(fadd_rounded)

  // If we fall through to here, then we need to round up, and also check if we
  // need to round to even. This occurs if all the bits of y's mantissa shifted
  // off the bottom are zero except for the round bit.
  //
  // Some of those bits are in r5 (the 32-bit version of the sum's mantissa).
  // It's cheap to check those, and should exclude _most_ cases where
  // round-to-even isn't needed.
  ADDS    r0, r0, #1          // simple round up
  LSLS    r5, r5, #(32-7)     // check top 7 bits
  BEQ     LOCAL_LABEL(fadd_check_rte)      // if those are zero, go to full RTE check
LOCAL_LABEL(fadd_rounded):
  // Put the sign+exponent back on. The leading bit of the mantissa increments
  // the exponent field unwantedly, so we must decrement r6 first to compensate
  // for that.
  SUBS    r6, r6, #1
  LSLS    r6, r6, #23
  ADDS    r0, r0, r6
  // If we haven't overflowed, it's now safe to return.
  CMP     r2, #255
  BGE     LOCAL_LABEL(fadd_overflow)
  POP     {r4,r5,r6,pc}

LOCAL_LABEL(fadd_overflow):
  // We have overflow, so we need to return an infinity of the correct sign. r0
  // already has the correct sign and exponent, so all we need to do is clear
  // its mantissa.
  LSRS    r0, r0, #23
  LSLS    r0, r0, #23
  POP     {r4,r5,r6,pc}

LOCAL_LABEL(fadd_sub):
  // We come here when fadd discovered it needed to subtract. Negate the second
  // operand and cross-jump into fsub.
  //
  // The cross-jump is done using BL, for greater branch range. That clobbers
  // lr, but that's OK, we weren't keeping anything in it at this point.
  EORS    r1, r1, r5
  BL      LOCAL_LABEL(fsub_magnitude)

LOCAL_LABEL(fadd_carry):
  // We come here if we carried a 1 bit off the top of r5 where we computed the
  // sum's mantissa. Shift back down by one and put a 1 bit in at the top.
  //
  // That would be easy with the RRX instruction from general AArch32, but we
  // don't have that here. Instead we OR in a 1 at the bottom, and move it to
  // the top by rotating right.
  //
  // A danger of shifting r5 down by a bit is that we lose the bit at the very
  // bottom, which might be important if it's the only nonzero bit below the
  // output mantissa, because then it determines whether we do RTE or not.
  // Fortunately, another copy of the same bit is still at the bottom of r4
  // (the shifted version of y's mantissa which we added to x's to make the
  // version of r5 _before_ we shifted it down). So the full RTE check will
  // have to remember to check that bit.
  MOVS    r0, #1
  ORRS    r5, r5, r0         // set low bit of r5
  RORS    r5, r5, r0         // and rotate right so that's now the high bit

  // Carrying off the top of the mantissa means that the output exponent must
  // be increased by 1. Increment both copies: the exponent by itself in r2
  // (used for overflow checking) and the exponent + sign in r6.
  ADDS    r2, r2, #1
  ADDS    r6, r6, #1

  // Now go back to the common code path for rounding and overflow checking.
  B       LOCAL_LABEL(fadd_renormed)

LOCAL_LABEL(fadd_check_rte):
  // We come here to do the full (and therefore expensive) check for round-to-
  // even: is our output number exactly on a rounding boundary, half way
  // between two representable numbers? That is, of the bits _not_ included in
  // the output mantissa, is the topmost bit 1 and all the rest 0?
  //
  // We only come here at all if we have already rounded the number up. So we
  // already know the topmost one of the lost bits is 1, and all we have to
  // check is whether the rest are 0.
  //
  // Also, we've already checked all the bits that were still in the 32-bit
  // version of the output mantissa, so we don't need to check those again ...
  //
  // ... well, _nearly_ all, because in the fadd_carry case, we shifted r5 down
  // by a bit _before_ that check. So we do need to re-check that one bit.
  //
  // The basic strategy is: r4 still contains the version of y's mantissa that
  // we shifted down before adding it to x. And r1 contains more or less the
  // original version of all of y, including the same mantissa. So if we shift
  // r4 back up again and XOR it with r1, we clear all the bits that we've
  // already checked, and leave only the ones we haven't.

  // Start by deliberately throwing away the low bit of r4, in case that
  // corresponded to the bit we lost off the bottom of r5 in fadd_carry. This
  // means we won't clear it in the XOR, and therefore, _will_ check it.
  LSRS    r4, r4, #1

  // Shift r4 back up by the same amount we shifted it down, and shift r1 to
  // the corresponding position, so that we can XOR them. The most convenient
  // way to do this is not to modify the variable shift count in r3, and
  // compensate for it by selecting the shift of r1 appropriately.
  //
  // As it happens, we end up with the implicit leading 1 bit of the mantissa
  // in bit 30 of the result - or rather, it would be if we'd set it, which in
  // r1 we haven't, because that's still the whole original input float.
  LSLS    r4, r4, r3
  LSLS    r1, r1, #7
  EORS    r1, r1, r4

  // But r1 wasn't just the mantissa of y; it also had the exponent, and its
  // leading bit was implicit. So the topmost two bits of r1 are useless: in r1
  // they're part of the exponent field. Exclude them from consideration.
  //
  // This doesn't lead to dropping any bit we really care about, because we're
  // never interested in the actual leading 1 bit of y's mantissa for round-to-
  // even purposes. Why not? Because we already know the round bit (the one
  // just off the bottom of the output mantissa) is a 1, which must have come
  // from y (it's too low down to come from x), and we only care about checking
  // all the bits below _that_. So y's leading 1 must be at least as high up as
  // the round bit, and therefore, isn't one of the bits we currently need to
  // check.
  LSLS    r1, r1, #2

  // Now if all those bits are zero, we're rounding to even. If _not_, we're
  // finished rounding, so go back to fadd_rounded to continue the main code
  // path.
  BNE     LOCAL_LABEL(fadd_rounded)

  // Clear the low bit of the output (rounding to even) and go back to the main
  // code path.
  MOVS    r4, #1
  BICS    r0, r0, r4
  B       LOCAL_LABEL(fadd_rounded)

LOCAL_LABEL(fadd_naninf):
  // We come here if at least one input is a NaN or infinity. If either or both
  // inputs are NaN then we hand off to __fnan2 which will propagate a NaN from
  // the input.
  //
  // On entry, we know r5 = 0x80000000 from the initial uncommon check. Also,
  // we already extracted the exponents of x and y into r2 and r3.
  ASRS    r4, r5, #7    // so r4 = 0xFF000000
  LSLS    r6, r0, #1    // r6 > r4 iff x is NaN
  CMP     r6, r4
  BHI     LOCAL_LABEL(fadd_nan)
  LSLS    r6, r1, #1    // r6 > r4 iff y is NaN
  CMP     r6, r4
  BHI     LOCAL_LABEL(fadd_nan)

  // No NaNs, so we have at least one infinity. Almost all additions involving
  // an infinity return the input infinity unchanged. The only exception is if
  // there are two infinities that have opposite signs (which can happen even
  // inf fadd, since on this code path we haven't cross-jumped into fsub),
  // where we return NaN.
  CMP     r2, r3        // at least one exponent is 0xFF, so if EQ, both are
  BEQ     LOCAL_LABEL(fadd_infinf)   //   and therefore we're adding infinity to infinity

  // With one infinity, we just find which register it's in, and return it.
  CMP     r2, #255
  BEQ     LOCAL_LABEL(fadd_ret_exact)  // just return x
LOCAL_LABEL(fadd_retb): // we reuse this code in the denormal handler
  MOVS    r0, r1          // otherwise, return y
LOCAL_LABEL(fadd_ret_exact):
  POP     {r4,r5,r6,pc}

LOCAL_LABEL(fadd_infinf):
  // With two infinities, we must check their relative sign. If they're the
  // same sign, we have no problem.
  MOVS    r4, r0
  EORS    r4, r4, r1
  BPL     LOCAL_LABEL(fadd_ret_exact)  // identical infinities, so just return one

  // But if we're adding two infinities of opposite sign, make a default quiet
  // NaN and return that.
  LDR     r0, =0x7fc00000
  POP     {r4,r5,r6,pc}

LOCAL_LABEL(fadd_nan):
  BL      SYMBOL_NAME(__fnan2)
  POP     {r4,r5,r6,pc}

LOCAL_LABEL(fadd_zerodenorm_x):
  // We come here if we found x was 0 or a denormal. We haven't set up r3 as
  // the exponent of y yet.
  LSRS    r3, r3, #24

  // Also, we checked for zero/denorm before checking for infinities and NaNs.
  // We know x isn't an infinity or NaN, but we must check y.
  CMP     r3, #255
  BEQ     LOCAL_LABEL(fadd_naninf)

  // Fall through to the next section. This repeats a pointless check for x
  // being NaN or infinity, but it would cost more cycles to branch round it.

LOCAL_LABEL(fadd_zerodenorm_y):
  // We come here if we found y was 0 or a denormal, but also by falling
  // through from above. So we may not yet have checked x for infinity/NaN. But
  // we have checked that y isn't.
  CMP     r2, #255
  BEQ     LOCAL_LABEL(fadd_naninf)

  // Now at least one of x,y is zero or denormal, and neither is infinite or
  // NaN. We haven't yet checked the signs and cross-jumped to fsub, but we can
  // handle all the zero cases without having to:
  //
  //  - if x = -y (including both being zero), return 0 of the appropriate sign
  //  - if x = 0, return y (including the case of same-signed zeroes)
  //  - if y = 0, return x
  SUBS    r6, r0, r1     // are x and y equal
  CMP     r6, r5         //   except for opposite sign bits? (r5 = 0x80000000)
  BEQ     LOCAL_LABEL(fadd_diffsame)
  LSLS    r6, r1, #1     // is y zero?
  BEQ     LOCAL_LABEL(fadd_ret_exact) // if so, return x
  LSLS    r6, r0, #1     // is x zero?
  BEQ     LOCAL_LABEL(fadd_retb)      // if so, return y

  // Now we've dealt with all the possibilities involving zeroes, so we have
  // either one denormal or two denormals. These cases are harder, and we don't
  // want to handle both signs at once, so check the signs and cross-branch
  // into fsub if they're different.
  MOVS    r6, r1
  EORS    r6, r6, r0
  BPL     LOCAL_LABEL(fadd_denorm)
  EORS    r1, r1, r5
  BL      LOCAL_LABEL(fsub_denorm)
LOCAL_LABEL(fadd_denorm):
  // Sort the operands into magnitude order. Now we know they have the same
  // sign, unsigned comparison is good enough for that.
  SUBS    r6, r0, r1
  BHS     LOCAL_LABEL(fadd_denorm_noswap)
  SUBS    r0, r0, r6
  ADDS    r1, r1, r6
LOCAL_LABEL(fadd_denorm_noswap):

  // We know one exponent is 0, so check if the other is too. We do this by
  // adding the two exponents together, achieving two things in one
  // instruction: it gets the nonzero exponent (if any) into r2 (saving us
  // swapping r2 with r3 in the sorting step above), and it sets Z if both were
  // zero.
  ADDS    r2, r2, r3
  BEQ     LOCAL_LABEL(fadd_denorm2)

  // Now exactly one operand is denormal, and it's y. We must go back to
  // fadd_doadd with all the registers appropriately set up.
  LSRS    r6, r0, #23  // r6 == sign and exponent of x
  LSLS    r4, r1, #8   // r4 == mantissa of y, with leading bit clear
  LSLS    r0, r0, #8
  ORRS    r0, r0, r5   // set high bit on mantissa of x
  SUBS    r3, r2, #1   // denormals are shifted as if they had exponent 1
  B       LOCAL_LABEL(fadd_doadd)

LOCAL_LABEL(fadd_diffsame):
  // Here we only support round-to-nearest mode, so the difference of two
  // identical things always returns +0.
  MOVS    r0, #0
  POP     {r4,r5,r6,pc}

LOCAL_LABEL(fadd_denorm2):
  // Here, x,y are both denormal, and we know we're doing magnitude addition.
  // So we can add the mantissas like ordinary integers, and if they carry into
  // the exponent, that's still the correct answer. But we have to avoid adding
  // two copies of the sign bit, so we clear that from y first.
  BICS    r1, r1, r5  // clear sign bit of y
  ADDS    r0, r0, r1  // add mantissas
  POP     {r4,r5,r6,pc}
END_COMPILERRT_FUNCTION(__addsf3)

DEFINE_COMPILERRT_THUMB_FUNCTION(__aeabi_frsub)
  // Reversed subtraction, that is, compute y-x, where x is in r0 and y in r1.
  //
  // We could implement this by simply swapping r0 with r1. But the point of
  // having a reversed-subtract in the first place is to avoid the caller
  // having to do that, so if we do it ourselves, it wastes all the time they
  // saved. So instead, on the fast path, we redo the sign check our own way
  // and branch to fadd_magnitude or fsub_magnitude.

  PUSH {r4,r5,r6,lr}

  MOVS    r5, #1
  LSLS    r5, r5, #31 // all cross-branches will expect to have r5 = 0x80000000

  // Extract the exponents and test for uncommon values. Note that we do the
  // zero/denormal tests the opposite way round from fsub, because we swap the
  // operands before branching to the corresponding fsub code, so this way our
  // first branch will enter fsub with the first of _its_ operands checked.
  LSLS    r2, r0, #1
  LSLS    r3, r1, #1
  LSRS    r3, r3, #24
  BEQ     LOCAL_LABEL(frsb_zerodenorm_y)
  LSRS    r2, r2, #24
  BEQ     LOCAL_LABEL(frsb_zerodenorm_x)
  CMP     r2, #255
  BEQ     LOCAL_LABEL(frsb_naninf)
  CMP     r3, #255
  BEQ     LOCAL_LABEL(frsb_naninf)

  // Decide which of fadd_magnitude and fsub_magnitude to branch to, and do so.
  EORS    r0, r0, r5
  MOVS    r4, r0
  EORS    r4, r4, r1
  BPL     LOCAL_LABEL(frsb_add)
  EORS    r1, r1, r5
  BL      LOCAL_LABEL(fsub_magnitude)
LOCAL_LABEL(frsb_add):
  BL      LOCAL_LABEL(fadd_magnitude)

  // Any uncommon operands to frsub are handled by just swapping the two
  // operands and going to fsub's handler. We're off the main fast path now, so
  // there's no need to try to optimise it any harder.
LOCAL_LABEL(frsb_zerodenorm_y):
  PUSH    {r0,r2}
  PUSH    {r1,r3}
  POP     {r0,r2}
  POP     {r1,r3}
  BL      LOCAL_LABEL(fsub_zerodenorm_x)  // we just swapped x and y, so now x is 0/denorm
LOCAL_LABEL(frsb_zerodenorm_x):
  PUSH    {r0,r2}
  PUSH    {r1,r3}
  POP     {r0,r2}
  POP     {r1,r3}
  BL      LOCAL_LABEL(fsub_zerodenorm_y)  // similarly, now we know y is
LOCAL_LABEL(frsb_naninf):
  PUSH    {r0,r2}
  PUSH    {r1,r3}
  POP     {r0,r2}
  POP     {r1,r3}
  BL      LOCAL_LABEL(fsub_naninf)
END_COMPILERRT_FUNCTION(__aeabi_frsub)

DEFINE_AEABI_FUNCTION_ALIAS(__aeabi_fsub, __subsf3)

DEFINE_COMPILERRT_THUMB_FUNCTION(__subsf3)
  // Main entry point for subtraction.
  PUSH {r4,r5,r6,lr}

  MOVS    r5, #1
  LSLS    r5, r5, #31

  // Extract the exponents into r2 and r3 and test for all uncommon values,
  // similarly to fadd.
  LSLS    r2, r0, #1
  LSLS    r3, r1, #1
  LSRS    r2, r2, #24
  BEQ     LOCAL_LABEL(fsub_zerodenorm_x)
  LSRS    r3, r3, #24
  BEQ     LOCAL_LABEL(fsub_zerodenorm_y)
  CMP     r2, #255
  BEQ     LOCAL_LABEL(fsub_naninf)
  CMP     r3, #255
  BEQ     LOCAL_LABEL(fsub_naninf)

  // Check the signs, and if they're unequal, cross-jump into fadd to do
  // magnitude addition. (Now we've excluded NaNs, it's safe to flip the sign
  // of y.)
  MOVS    r4, r0
  EORS    r4, r4, r1
  BMI     LOCAL_LABEL(fsub_add)
LOCAL_LABEL(fsub_magnitude):
  // If we get here, we're subtracting operands with equal signs (i.e. a
  // magnitude subtraction). First thing to do is put operands in magnitude
  // order, so that x >= y. However, if they are swapped, we must also negate
  // both of them, since A - B = (-B) - (-A).
  SUBS    r4, r0, r1
  BHS     LOCAL_LABEL(fsub_swapped)
  EORS    r4, r4, r5
  SUBS    r0, r0, r4
  ADDS    r1, r1, r4
  // We must also swap the pre-extracted exponents here.
  EORS    r2, r2, r3
  EORS    r3, r3, r2
  EORS    r2, r2, r3
LOCAL_LABEL(fsub_swapped):
  // Save the sign and exponent of the larger operand to use for the result (up
  // to renormalisation), and calculate the exponent difference for shifting
  // one mantissa relative to the other.
  LSRS    r6, r0, #23
  SUBS    r3, r2, r3

  // Shift the mantissas up to the top of the words. In the process we put y's
  // shifted mantissa into a separate register, keeping the original for later
  // reference. Also, although we set the leading bit of y, we _clear_ the
  // leading bit of x, which is just as quick and saves us having to decrement
  // the output exponent later to compensate.
  LSLS    r0, r0, #8
  LSLS    r4, r1, #8
  BICS    r0, r0, r5
  ORRS    r4, r4, r5

LOCAL_LABEL(fsub_dosub): // we may come back here after sorting out denorms

  // We get here with:
  //   Operands known to be numeric rather than zero/infinity/NaN;
  //   r0 = mantissa of larger operand (in top 24 bits, with high bit clear)
  //   r4 = mantissa of smaller operand (in top 24 bits, with high bit set)
  //   r1 = original smaller operand (up to maybe a sign flip)
  //   r6 = result sign/exponent (in low 9 bits)
  //   r2 = plain result exponent (in low 8 bits, i.e. r6 & 0xFF)
  //   r3 = exponent difference.
  //
  // Begin calculating the output mantissa by shifting y's mantissa right and
  // subtracting. This may leave the mantissa too large by one, if the bits
  // shifted out of y are nonzero. We correct this during rounding if
  // necessary.
  LSRS    r4, r4, r3
  SUBS    r5, r0, r4

  // This may have cleared the high bit of the output mantissa, in which case
  // we must renormalise. Our strategy is to split into three code paths, on
  // two of which an awkward case is known not to arise:
  //  * no need to renormalise at all => underflow can't happen
  //  * shift up by exactly 1 bit
  //  * shift up by more than 1 bit => rounding can't happen (result is exact)
  //
  // First branch out of line for the first case, which we can detect because
  // the N flag tells us whether the top mantissa bit is still set.
  BPL     LOCAL_LABEL(fsub_renormed)

  // Renormalise by one bit, and check the new top bit to see if we need to
  // renormalise by more than that.
  LSLS    r5, r5, #1
  BPL     LOCAL_LABEL(fsub_renorm_big) // if new top bit still clear, renormalise by more
  // Decrement both exponent registers (r6 with the sign, r2 without). We
  // decrement r6 by 2 instead of 1, because now the output mantissa has the
  // top bit set, so we must compensate when we put the sign and exponent back
  // on.
  //
  // The extra decrement of r6 might carry into the sign bit. This doesn't
  // matter on the fast path, because the leading bit in the mantissa will undo
  // it. But we need to account for it in the underflow handler for this path.
  SUBS    r6, r6, #2
  SUBS    r2, r2, #1
  // The decrement of the pure exponent value also doubles as a check for
  // underflow, because we underflowed precisely if the exponent went to 0.
  BEQ     LOCAL_LABEL(fsub_underflow_1)
LOCAL_LABEL(fsub_renormed):
  // Now we have the output mantissa in r5. It may or may not have the high bit
  // set, depending on which branch of the code we've come through. But r6 has
  // been adjusted appropriately, so that we can make a basically right output
  // value (before rounding) by adding r6 << 23 to r5 >> 8.
  //
  // If any nonzero bits were shifted off the bottom of y, then the true value
  // of the output mantissa might be slightly _less_ than the value in r5.
  // However the maximum difference is about 2^{-7} ULP relative to the final
  // result (because it's at most one ULP of the 32-bit output mantissa in r5).
  // So it doesn't affect the result in round-to-nearest mode unless it puts us
  // just below a rounding boundary, which means we can ignore it until the
  // full round-to-even check.
  LSLS    r6, r6, #23  // prepare sign and exponent
  LSRS    r0, r5, #8   // shift down, and put the round bit into C
  BCS     LOCAL_LABEL(fsub_round)   // diverge based on round bit
  // If the round bit shifted off the bottom of r5 was clear, then we're not
  // rounding up, so we can make the output value and finish immediately.
  ADDS    r0, r0, r6   // reconstitute output value without rounding
  POP     {r4,r5,r6,pc}
LOCAL_LABEL(fsub_round):
  // Otherwise, we're rounding, in three stages. First round up; then cheaply
  // check the low bits of r5 (the 32-bit version of the mantissa) so that we
  // can rule out round-to-even if any of those is nonzero; finally, in as few
  // cases as possible, check the rest of y's mantissa to check for RTE fully.
  ADCS    r0, r0, r6      // reconstitute output value while rounding up
  LSLS    r5, r5, #(32-7) // check first 7 guard bits
  BEQ     LOCAL_LABEL(fsub_check_rte)  // if the're all 0, do the full check for RTE
  POP     {r4,r5,r6,pc}   // otherwise we're done

LOCAL_LABEL(fsub_add):
  // Trampoline to cross-jump to fadd, because a 16-bit branch won't reach that
  // far. Also a convenient place to flip y's sign, so we only have to do it
  // once.
  EORS    r1, r1, r5      // we know r5 = 0x80000000
  BL      LOCAL_LABEL(fadd_magnitude)  // clobbers lr, which doesn't matter

LOCAL_LABEL(fsub_check_rte):
  // Full check for round-to-even, in the same style as fadd_check_rte: r4
  // still contains the version of y's mantissa that we shifted down before
  // subtracting from x, and r1 contains the original version of that mantissa.
  // So if we shift r4 back up again and XOR it with r1, we clear all the bits
  // that we've already checked, and leave only the ones we haven't. The only
  // exception is the leading mantissa bit, which is implicit in r1, but this
  // can never affect round-to-even, because if we rounded at all then the
  // round bit must have come from y, so the leading bit of y is at the round
  // bit or above, hence not one of the bits we're checking for RTE.
  LSLS    r4, r4, r3  // undo the shift of y's mantissa
  LSLS    r1, r1, #8  // shift y's original mantissa back to the same place
  EORS    r1, r1, r4  // find any differences
  LSLS    r1, r1, #1  // but ignore the leading mantissa bit
  BEQ     LOCAL_LABEL(fsub_rte)    // if all bits now clear, we're rounding to even

  // If we're not RTEing, we must undo the simplistic rounding we've already
  // done. (We incremented the result based on the belief that the shifted-off
  // data started 0x80xxx, but it turns out that xxx is slightly negative, so
  // actually we had 0x7Fyyy.)
  SUBS    r0, r0, #1
  POP     {r4,r5,r6,pc}
LOCAL_LABEL(fsub_rte):
  // Actually round to even, by clearing the low bit of the output.
  MOVS    r4, #1
  BICS    r0, r0, r4
  POP     {r4,r5,r6,pc}

LOCAL_LABEL(fsub_renorm_big):
  // Now we know that we must renormalise by at least 2 bits, which may also
  // give a denormal or zero result.
  //
  // This means no rounding can possibly be needed: if the subtraction cleared
  // the top two bits of the mantissa, it means we computed A-B and found it
  // was less than A/2, so B > A/2, so the exponent difference was at most 1.
  // Hence the result mantissa fits in 24 bits even before renormalisation, and
  // the top bit is clear, so it fits in 23 bits, i.e. it is exact.

  // Detect an actual zero result, and go and return it.
  BEQ     LOCAL_LABEL(fsub_diffsame)

  // Renormalise by binary search. (16-bit Thumb has no CLZ instruction.) We'll
  // accumulate the total exponent adjustment in r0. It starts at 1 rather than
  // 0, because we've shifted the mantissa left by one bit already.
  MOVS    r0, #1

  // If the top 16 bits of r5 are clear, shift up by 16 and adjust r0 to match.
  LSRS    r3, r5, #(32-16)
  BNE     LOCAL_LABEL(fsub_denorm_noshift16)
  LSLS    r5, r5, #16
  ADDS    r0, r0, #16
LOCAL_LABEL(fsub_denorm_noshift16):
  // Same for 8 bits
  LSRS    r3, r5, #(32-8)
  BNE     LOCAL_LABEL(fsub_denorm_noshift8)
  LSLS    r5, r5, #8
  ADDS    r0, r0, #8
LOCAL_LABEL(fsub_denorm_noshift8):
  // 4 bits
  LSRS    r3, r5, #(32-4)
  BNE     LOCAL_LABEL(fsub_denorm_noshift4)
  LSLS    r5, r5, #4
  ADDS    r0, r0, #4
LOCAL_LABEL(fsub_denorm_noshift4):
  // 2 bits
  LSRS    r3, r5, #(32-2)
  BNE     LOCAL_LABEL(fsub_denorm_noshift2)
  LSLS    r5, r5, #2
  ADDS    r0, r0, #2
LOCAL_LABEL(fsub_denorm_noshift2):
  // 1 bit
  LSRS    r3, r5, #(32-1)
  BNE     LOCAL_LABEL(fsub_denorm_noshift1)
  LSLS    r5, r5, #1
  ADDS    r0, r0, #1
LOCAL_LABEL(fsub_denorm_noshift1):

  // Update our two copies of the exponent (with sign in r6, without in r2).
  SUBS    r6, r6, r0
  SUBS    r2, r2, r0
  // Shift the mantissa and exponent into the right places to combine them.
  LSLS    r4, r5, #1              // clear leading bit of mantissa
  LSRS    r0, r4, #9              // and shift it down
  LSLS    r4, r6, #23             // shift sign and exponent up
  ADDS    r0, r0, r4              // put them together
  // Check for underflow, which occurs if the output exponent is less than 1
  // (including having gone negative).
  CMP     r2, #1
  BLT     LOCAL_LABEL(fsub_underflow_2)
  POP     {r4,r5,r6,pc}

LOCAL_LABEL(fsub_diffsame):
  // Here we only support round-to-nearest mode, so the difference of two
  // identical things always returns +0.
  MOVS    r0, #0
  POP     {r4,r5,r6,pc}

LOCAL_LABEL(fsub_underflow_1):
  // We come here if renormalising by one bit reduced the output exponent to
  // zero. In other words, the output value in x is denormal (hence exact) and
  // wants shifting down by exactly 9 bits (8 bits of exponent plus the bit we
  // already shifted it by), and then the sign bit putting back on.
  //
  // Also, before we get the sign bit from r6, we must add 1 to it, because of
  // the possibility that decrementing it carried into the sign bit.
  ADDS    r6, r6, #1    // undo potential sign-flipping carry
  LSRS    r6, r6, #8    // isolate the sign bit
  LSLS    r6, r6, #31   // and shift it up to the top
  LSRS    r0, r5, #9    // construct the output mantissa
  ORRS    r0, r0, r6    // and combine with the sign bit
  POP     {r4,r5,r6,pc}

LOCAL_LABEL(fsub_underflow_2):
  // We come here if multi-bit renormalisation found a denormal. The mantissa
  // has its leading bit set at the top of r5, so it needs shifting down 8 bits
  // to where it would be in a normalised number, and then further: if the
  // output exponent is 0 (meaning the exponent just below a normalised number)
  // then we shift one extra bit, if it's -1 then we shift two extra bits, and
  // so on. So in total we shift down by 8 + (1 - exp) = 9 - exp.
  RSBS    r4, r6, #0
  ADDS    r4, r4, #9
  LSRS    r5, r5, r4    // shift mantissa into place

  // Extract the sign bit from r6 and combine it with that denormal. r6 could
  // be 0 or could be negative, so we must add enough to it to make it reliably
  // positive. Any offset that works is fine; we'll use 0xc0, which is the
  // offset used by IEEE 754:1985 underflow intermediate values.
  ADDS    r6, r6, #0xc0 // rebias to correct sign bit
  LSRS    r6, r6, #8    // isolate the sign bit
  LSLS    r0, r6, #31   // and shift it up to the top
  ADDS    r0, r0, r5    // combine with the denormalised mantissa
  POP     {r4,r5,r6,pc}

LOCAL_LABEL(fsub_naninf):
  // We come here if at least one input is a NaN or infinity. If either or both
  // inputs are NaN then we hand off to __fnan2 which will propagate a NaN from
  // the input.
  // We come here if at least one of x,y is a NaN or infinity.
  // Their exponents are reliably always in r2 and r3
  // respectively.
  ASRS    r4, r5, #7    // so r4 = 0xFF000000
  LSLS    r6, r0, #1    // r6 > r4 iff x is NaN
  CMP     r6, r4
  BHI     LOCAL_LABEL(fsub_nan)
  LSLS    r6, r1, #1    // r6 > r4 iff y is NaN
  CMP     r6, r4
  BHI     LOCAL_LABEL(fsub_nan)

  // No NaNs, so we have at least one infinity. Almost all additions involving
  // an infinity return the input infinity unchanged. The only exception is
  // subtracting two infinities that have the same sign, where we return NaN.
  CMP     r2, r3        // at least one exponent is 0xFF, so if EQ, both are
  BEQ     LOCAL_LABEL(fsub_infinf)

  // If x is infinite and y is finite, return x.
  CMP     r2, #255
  BEQ     LOCAL_LABEL(fsub_ret_exact)
LOCAL_LABEL(fsub_retminusy):
  // If x is finite and y is infinite, return -y.
  MOVS    r0, r1
  EORS    r0, r0, r5    // negate y
LOCAL_LABEL(fsub_retx):
LOCAL_LABEL(fsub_ret_exact):
  POP     {r4,r5,r6,pc}
LOCAL_LABEL(fsub_infinf):
  // With two infinities, we must check their relative sign. If they have
  // opposite sign, we just return x (which is the one with the same sign as
  // the output).
  MOVS    r4, r0
  EORS    r4, r4, r1
  BMI     LOCAL_LABEL(fsub_ret_exact)

  // But if we're subtracting two infinities of the same sign, make a default
  // quiet NaN and return that.
  LDR     r0, =0x7fc00000
  POP     {r4,r5,r6,pc}

LOCAL_LABEL(fsub_nan):
  BL      SYMBOL_NAME(__fnan2)
  POP     {r4,r5,r6,pc}

LOCAL_LABEL(fsub_zerodenorm_x):
  // We come here if we found x was 0 or a denormal. We haven't set up r3 as
  // the exponent of y yet.
  LSRS    r3, r3, #24

  // Also, we checked for zero/denorm before checking for infinities and NaNs.
  // We know x isn't an infinity or NaN, but we must check y.
  CMP     r3, #255
  BEQ     LOCAL_LABEL(fsub_naninf)

  // Fall through to the next section. This repeats a pointless check for x
  // being NaN or infinity, but it would cost more cycles to branch round it.

LOCAL_LABEL(fsub_zerodenorm_y):
  // We come here if we found y was 0 or a denormal, but also by falling
  // through from above. So we may not yet have checked x for infinity/NaN. But
  // we have checked that y isn't.
  CMP     r2, #255
  BEQ     LOCAL_LABEL(fsub_naninf)

  // Now at least one of x,y is zero or denormal, and neither is infinite or
  // NaN. We haven't yet checked the signs and cross-jumped to fsub, but we can
  // handle all the zero cases without having to:
  //
  //  - if x = -y (including both being zero), return 0 of the appropriate sign
  //  - if y = 0, return x (including the case of oppositely signed zeroes)
  //  - if x = 0 and y != 0, return -y
  CMP     r0, r1         // are x and y equal?
  BEQ     LOCAL_LABEL(fsub_diffsame)
  LSLS    r6, r1, #1     // is y zero?
  BEQ     LOCAL_LABEL(fsub_retx)      // if so, return x
  LSLS    r6, r0, #1     // is x zero?
  BEQ     LOCAL_LABEL(fsub_retminusy) // if so, return -y

  // Now we've dealt with all the possibilities involving zeroes, so we have
  // either one denormal or two denormals. These cases are harder, and we don't
  // want to handle both signs at once, so check the signs and cross-branch
  // into fadd if they're different.
  MOVS    r6, r1
  EORS    r6, r6, r0
  BPL     LOCAL_LABEL(fsub_denorm)
  EORS    r1, r1, r5
  BL      LOCAL_LABEL(fadd_denorm)
LOCAL_LABEL(fsub_denorm):
  // Sort the operands into magnitude order. Now we know they have the same
  // sign, unsigned comparison is good enough for that.
  SUBS    r6, r0, r1
  BHS     LOCAL_LABEL(fsub_denorm_noswap)
  EORS    r6, r6, r5              // flip the signs in the process
  SUBS    r0, r0, r6
  ADDS    r1, r1, r6
LOCAL_LABEL(fsub_denorm_noswap):

  // We know one exponent is 0, so check if the other is too. We do this by
  // adding the two exponents together, achieving two things in one
  // instruction: it gets the nonzero exponent (if any) into r2 (saving us
  // swapping r2 with r3 in the sorting step above), and it sets Z if both were
  // zero.
  ADDS    r2, r2, r3
  BEQ     LOCAL_LABEL(fsub_denorm2)

  // Now exactly one operand is denormal, and it's y. We must go back to
  // fsub_dosub with all the registers appropriately set up.
  LSRS    r6, r0, #23  // r6 == sign and exponent of x
  LSLS    r4, r1, #8   // r4 == mantissa of y, with leading bit clear
  LSLS    r0, r0, #8
  BICS    r0, r0, r5   // clear high bit on mantissa of x
  SUBS    r3, r2, #1   // denormals are shifted as if they had exponent 1
  B       LOCAL_LABEL(fsub_dosub)

LOCAL_LABEL(fsub_denorm2):
  // Here, x,y are both denormal, and we know we're doing magnitude addition.
  // So we can subtract the mantissas like ordinary integers. But we have to
  // avoid subtracting y's sign bit from x's.
  BICS    r1, r1, r5  // clear sign bit of y
  SUBS    r0, r0, r1  // subtract mantissas
  POP     {r4,r5,r6,pc}
END_COMPILERRT_FUNCTION(__subsf3)

NO_EXEC_STACK_DIRECTIVE

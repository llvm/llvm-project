//===-- addsf3.S - Add/subtract single precision floating point numbers ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __addsf3 and __subsf3 functions (single precision
// floating point number addition and subtraction), with the IEEE-754 default
// rounding (to nearest, ties to even), for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

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
// subtracted. So fadd jumps across into the middle of fsub if it sees that the
// signs are different, and vice versa. Then the main code path in fadd handles
// magnitude addition, and the one in fsub handles magnitude subtraction.
//
// NaNs are checked first, so that an input NaN can be propagated exactly,
// including its sign bit. After ruling out that case, it's safe to flip the
// sign of one of the inputs, so that during the cross-calls, x - y can be
// rewritten as x + (-y) and vice versa.

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__addsf3)
  push {r4, lr}
  vmov r0, s0
  vmov r1, s1
  bl __aeabi_fadd
  vmov s0, r0
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__addsf3, __aeabi_fadd)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_fadd)
  // Test for all uncommon values at once: infinities, NaNs, denormals and
  // zeroes. Branch out of line if any are found. We do this by XORing each
  // input with itself shifted left by a bit, which means that exponents 00 and
  // FF will both end up with seven zero bits at the top.
  eor     r2, r0, r0, lsl #1   // combine x with itself shifted
  eor     r3, r1, r1, lsl #1   // same for y
  tst     r2, #0x7F000000      // is x uncommon?
  tstne   r3, #0x7F000000      // if not, is y uncommon?
  beq     LOCAL_LABEL(add_uncommon)        // if either, branch out of line

  // Now we have two normalised numbers. If their signs are opposite, we should
  // be subtracting their magnitudes rather than adding, so cross-jump to fsub.
  teq     r0, r1               // set N if signs are unequal
  eormi   r1, r1, #1 << 31     // if so, flip the sign of y
  bmi     LOCAL_LABEL(sub_magnitude)       // and go to magnitude subtraction
LOCAL_LABEL(add_magnitude):
  // If we get here, we're adding operands with equal signs (i.e. a magnitude
  // addition). First thing to do is put the operands in magnitude order, so
  // that x >= y.
  subs    r2, r0, r1           // compare inputs, also keeping x-y
  sublo   r0, r0, r2           // if x<y then turn x into y, using value in r2
  addlo   r1, r1, r2           // and similarly turn y into x

  // Keep the sign and exponent of the larger input, to use as the sign and
  // exponent of the output (up to carries and overflows). Also calculate the
  // exponent difference, which tells us how far we'll need to shift y's
  // mantissa right to add it to x's.
  //
  // The shifted-right values will include the sign bits as well as the
  // exponents, but that's OK, in this branch the two sign bits are the same,
  // so they'll cancel when subtracted.
  //
  // The exponent difference can be as large as 0xFE (maximum exponent minus
  // minimum), which still fits in 8 bits, so shifting right by that amount is
  // well defined in AArch32.
  mov     r2, r0, lsr #23
  sub     r3, r2, r1, lsr #23

  // Extract both mantissas, moved up to the top of the word, with the leading
  // 1 made explicit.
  mov     r12, #1 << 31        // the leading 1 by itself
  orr     r0, r12, r0, lsl #8
  orr     r1, r12, r1, lsl #8

LOCAL_LABEL(add_doadd):
  // Here we perform the actual addition. We either fell through from the code
  // above, or jumped back to here after handling an input denormal.
  //
  // We get here with:
  //   Operands known to be numeric rather than zero/infinity/NaN;
  //   r0 = mantissa of larger operand (in high 24 bits);
  //   r1 = mantissa of smaller operand (in high 24 bits);
  //   r2 = result sign and exponent (in low 9 bits);
  //   r3 = exponent difference.
  //
  // For normal inputs, the mantissa registers (r0,r1) will have the top bit
  // set. Denormals will leave that bit clear, treating the number as
  // 0.[mantissa] x 2^(fixed exponent) instead of renormalising to 1.[mantissa]
  // x 2^(variable exponent) as a multiplication would want.

  // Actually shift the smaller mantissa downwards and add them together.
#if !__thumb__
  adds    r12, r0, r1, lsr r3  // CS if x >= 2.0
#else
  // Thumb can't fold a register-controlled shift into an add, so we must use
  // two separate instructions.
  lsr     r12, r1, r3
  adds    r12, r0, r12
#endif

  // If that addition carried off the top of r12, then the number has increased
  // its exponent. Diverge into a completely separate code path for that case,
  // because there we must check for overflow.
  bcs     LOCAL_LABEL(add_carry)

  // Here, on the non-carrying path, we don't need to check for overflow at
  // all. If there is an overflow it can only be due to rounding up, so the
  // overflowed mantissa will be all zeroes, so the naively generated output
  // will look like the correct infinity anyway.
  //
  // We shift the mantissa down to its final position, and recombine it with
  // the sign + exponent (in r2) via addition. We keep the bit shifted off the
  // bottom of the mantissa in C, and then use ADC for the recombination, which
  // causes us to round up if that bit was set without needing an extra
  // instruction. But the leading bit of the mantissa increments the exponent
  // field unwantedly, so we must decrement r2 first to compensate for that.
  sub     r2, r2, #1
  movs    r0, r12, lsr #8
  adc     r0, r0, r2, lsl #23

  // If we _didn't_ round up, then we're done.
  bxcc    lr

  // But if we did round up, then we must also check if we need to round to
  // even. This occurs if all the bits of y's mantissa shifted off the bottom
  // are zero except for the round bit.
  //
  // Some of those bits are in r12 (the 32-bit version of the sum's mantissa).
  // It's cheap to check those, and should exclude _most_ cases where
  // round-to-even isn't needed.
  tst     r12, #127
  bxne    lr

  // Failing that, we have to go back to the original mantissa of y (still in
  // r1) and work out exactly how many bits of it to check.
  rsb     r3, r3, #32  // opposite of the amount we shifted y right by
  lsls    r1, r1, r3   // shift y left by that amount instead

  // Now if Z is set, we do round to even, which works by just clearing the low
  // bit of the output mantissa. This undoes the round-up if we rounded up to
  // an odd mantissa, and otherwise, makes no difference.
  biceq   r0, r0, #1

  // And now we're done.
  bx      lr

LOCAL_LABEL(add_carry):
  // This is the separate code path in which adding the mantissas together
  // caused a carry off the top of the word, so that the exponent of the output
  // incremented (even before rounding). Start by shifting the carry bit back
  // in.
  rrx     r0, r12

  // Now recombine the sign and exponent, and do the basic rounding (apart from
  // round to even), in the same way as the non-carrying code path above.
  // However this time we don't decrement r2, because we want our exponent to
  // come out bigger by 1 than in the other code path.
  movs    r0, r0, lsr #8       // shift mantissa down to the right position
  adc     r0, r0, r2, lsl #23  // recombine with sign+exponent, and round

  // Note that the mantissa cannot have overflowed during rounding: if it has
  // all bits 1 before rounding, both operands must also have had all mantissa
  // bits 1, and the same exponent - which implies the round bit was 0.
  //
  // So we definitely have the correct output exponent. There are two problems
  // left: we might need to round to even, and we might have overflowed.

  // First, do the cheap check that _usually_ rules out round-to-even. We only
  // do this if C is set (i.e. if we rounded up), and we end up with Z=0 if no
  // RTE. This relies on also having Z=0 already, in the case where we _didn't_
  // round up - and that must be true because the last time we set the flags it
  // was by shifting down the output mantissa, and that will always have had
  // its leading bit set.
  tstcs   r12, #255       // test one more bit than on the no-carry path

  // Now if Z=1 then we need to do the full check for RTE. But first, prepare a
  // version of the output value shifted left by 1 where it's convenient to
  // check its exponent for overflow. (We couldn't do that until we'd finished
  // with r12 by testing it in the previous instruction.)
  mov     r12, r0, lsl #1

  // Now, if we need to check for RTE, go off and do it.
  beq     LOCAL_LABEL(add_roundeven_ovf)

  // Otherwise, we still need to check for overflow.
  cmp     r12, #0xff000000  // if r12 >= this, the exponent has overflowed
  bxlo    lr                // so if not, we can leave
  b       LOCAL_LABEL(add_ovf)          // but if so, go and handle overflow

LOCAL_LABEL(add_roundeven_ovf):
  // We came here if we detected a need to do the full check for RTE. But we
  // may _also_ have overflowed, and just not have noticed yet.

  // Same round-to-even check as in the non-carry case above.
  rsb     r3, r3, #32  // opposite of the amount we shifted y right by
  lsls    r1, r1, r3   // shift y left by that amount instead
  biceq   r0, r0, #1   // and if the remaining bits are all 0, round to even

  // Now check for overflow, and if none, we're done.
  cmp     r12, #0xff000000  // if r12 >= this, the exponent has overflowed
  bxlo    lr                // so if not, we can leave

  // If we get here, we have definitely overflowed. Moreover, the exponent
  // field of the number is exactly 0xff. So all we have to do is clear the
  // mantissa, to make it into an infinity of the output sign.
LOCAL_LABEL(add_ovf):
  bfc     r0, #0, #23
  bx      lr

LOCAL_LABEL(add_uncommon):
  // We come here if the entry-point check says that at least one of x and y
  // has an uncommon (FF or 00) exponent. So we have at least one NaN,
  // infinity, denormal or zero, but we don't know which, or which operand it's
  // in. And we could have any combination of those types of input, in _both_
  // operands.

  // Detect FF exponents (NaNs or infinities) and branch again for those.
  mov     r12, #0xFF000000
  bics    r2, r12, r0, lsl #1
  bicsne  r2, r12, r1, lsl #1
  beq     LOCAL_LABEL(add_naninf)

  // Now we know both inputs are finite, but there may be denormals or zeroes.
  // So it's safe to do the same sign check and cross-jump as we did on the
  // fast path.
  teq     r0, r1             // opposite signs?
  eormi   r1, r1, #1 << 31   // if so, negate the second operand
  bmi     LOCAL_LABEL(sub_zerodenorm)    // and cross-jump to the fsub version of this code
LOCAL_LABEL(add_zerodenorm):
  // Now we know x and y have the same sign, and at least one of them is zero
  // or denormal. If there aren't any zeroes, we'll end up rejoining the fast
  // path, so we must set up all the same registers, and do our checks for zero
  // in line with that.
  //
  // Start by exactly repeating the initial fast-path setup code: sort into
  // magnitude order, get the output sign+exponent and the exponent shift.
  subs    r2, r0, r1           // compare inputs, also keeping x-y
  sublo   r0, r0, r2           // if x<y then turn x into y, using value in r2
  addlo   r1, r1, r2           // and similarly turn y into x
  mov     r2, r0, lsr #23      // get exponent of x (the sign bit will cancel)
  sub     r3, r2, r1, lsr #23  // subtract exponent of y to get shift count

  // Shift y's mantissa up to the top of r1. We know y has exponent 0 (at least
  // one of the inputs does, and we've sorted them by now). So we definitely
  // don't need to set the leading bit on y's mantissa; also, if r1 becomes
  // zero, then we know we have an addition to 0, and otherwise, we know both
  // inputs are nonzero.
  movs    r1, r1, lsl #8       // is y zero?
  bxeq    lr                   // if so, just return x

  // Now we know there aren't any zeroes, and that y is a denormal. x might or
  // might not be a denormal, so we must check that and decide whether to set
  // its top mantissa bit.
  mov     r0, r0, lsl #8       // shift mantissa of x to the top of r0
  tst     r2, #255             // is x's exponent 0? If so, it's denormal
  orrne   r0, r0, #1 << 31     // if not, set leading bit of x,
  subne   r3, r3, #1           //   adjust exponent difference,
  bne     LOCAL_LABEL(add_doadd)           //  and go back to mainstream

  // If both operands are denormals, addition becomes trivial: denormals and
  // the smallest exponent of normalised numbers both multiply the mantissa by
  // the same power of 2, so we can just add the mantissas together and put the
  // output sign back on.
  add     r0, r0, r1           // make the output mantissa
  mov     r0, r0, lsr #8       // shift it into position
  orr     r0, r0, r2, lsl #23  // put the sign back at the top
  bx      lr                   // done!

LOCAL_LABEL(add_naninf):
  // We come here if at least one input is a NaN or infinity. If either or both
  // inputs are NaN then we hand off to __fnan2 which will propagate a NaN from
  // the input.
  mov     r12, #0xFF000000
  cmp     r12, r0, lsl #1          // if (r0 << 1) > 0xFF000000, r0 is a NaN
  blo     SYMBOL_NAME(__compiler_rt_fnan2)
  cmp     r12, r1, lsl #1
  blo     SYMBOL_NAME(__compiler_rt_fnan2)

LOCAL_LABEL(add_inf):
  // No NaNs, so we have at least one infinity. Almost all additions involving
  // an infinity return the input infinity unchanged. The only exception is if
  // there are two infinities that have opposite signs (which can happen even
  // inf fadd, since on this code path we haven't cross-jumped into fsub),
  // where we return NaN.
  eor     r2, r0, r1               // see how the two inputs differ
  cmp     r2, #0x80000000          // +inf + -inf?
  subeq   r0, r2, #0x00400000      // if so, make the default output QNaN
  bxeq    lr                       // and return it
  cmp     r12, r0, lsl #1          // otherwise, is r0 the infinity?
  movne   r0, r1                   // no, so it's r1
  bx      lr                       // return the infinite input unchanged

END_COMPILERRT_FUNCTION(__aeabi_fadd)

DEFINE_COMPILERRT_FUNCTION(__aeabi_frsub)
  // Reversed subtraction, that is, compute y-x, where x is in r0 and y in r1.
  //
  // We could implement this by simply swapping r0 with r1. But the point of
  // having a reversed-subtract in the first place is to avoid the caller
  // having to do that, so if we do it ourselves, it wastes all the time they
  // saved. So instead, on the fast path, we redo the sign check our own way
  // and branch to fadd_magnitude or fsub_magnitude.

  // First rule out denormals and zeroes, using the same test as fadd and fsub.
  eor     r2, r0, r0, lsl #1
  eor     r3, r1, r1, lsl #1
  tst     r2, #0x7F000000
  tstne   r3, #0x7F000000
  beq     LOCAL_LABEL(rsb_uncommon)

  // Now we know we only have finite inputs, it's safe to implement the
  // reversal of the operand order by flipping signs. (Preserving the sign of
  // an input NaN was the only case where that wasn't right.)

  eor     r0, r0, #1 << 31  // flip sign of the operand we're subtracting
  teq     r0, r1            // are the signs now the same?
  bpl     LOCAL_LABEL(add_magnitude)    // if so, we're doing magnitude addition
  eor     r1, r1, #1 << 31  // otherwise, flip the other sign too
  b       LOCAL_LABEL(sub_magnitude)    // and we're doing magnitude subtraction

LOCAL_LABEL(rsb_uncommon):
  // Any uncommon operands to frsub are handled by just swapping the two
  // operands and going to fsub's handler. We're off the main fast path now, so
  // there's no need to try to optimise it any harder.
  eor     r0, r0, r1
  eor     r1, r1, r0
  eor     r0, r0, r1
  b       LOCAL_LABEL(sub_uncommon)

END_COMPILERRT_FUNCTION(__aeabi_frsub)

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__subsf3)
  push {r4, lr}
  vmov r0, s0
  vmov r1, s1
  bl __aeabi_fsub
  vmov s0, r0
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__subsf3, __aeabi_fsub)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_fsub)
  // Main entry point for subtraction.
  //
  // Start by testing for uncommon operands in the usual way.
  eor     r2, r0, r0, lsl #1
  eor     r3, r1, r1, lsl #1
  tst     r2, #0x7F000000
  tstne   r3, #0x7F000000
  beq     LOCAL_LABEL(sub_uncommon)

  // Check the signs, and if they're unequal, cross-jump into fadd to do
  // magnitude addition. (Now we've excluded NaNs, it's safe to flip the sign
  // of y.)
  teq     r0, r1
  eormi   r1, r1, #1 << 31
  bmi     LOCAL_LABEL(add_magnitude)

LOCAL_LABEL(sub_magnitude):
  // If we get here, we're subtracting operands with equal signs (i.e. a
  // magnitude subtraction). First thing to do is put operands in magnitude
  // order, so that x >= y. However, if they are swapped, we must also negate
  // both of them, since x - y = (-y) - (-x).
  subs    r2, r0, r1            // LO if we must swap the operands
#if !__thumb__
  // Conditional on LO, swap the operands, by adding/subtracting the difference
  // between them that we just wrote into r2. Negate them both in the process
  // by flipping the high bit of r2 first.
  eorlo   r2, r2, #1 << 31
  sublo   r0, r0, r2
  addlo   r1, r1, r2
#else
  // In Thumb, conditionally branch round these three instructions, instead of
  // conditionally executing them with an ITTT LO. Rationale: on the simpler
  // Thumb-only cores such as Cortex-M3, a branch only takes two cycles and an
  // IT costs one, so this saves two cycles in the untaken case and doesn't
  // impact the taken case at all.
  bhs     0f
  eor     r2, r2, #1 << 31
  sub     r0, r0, r2
  add     r1, r1, r2
0:
#endif

  // Save the sign and exponent of the larger operand to use for the result (up
  // to renormalisation), and calculate the exponent difference for shifting
  // one mantissa relative to the other.
  mov     r2, r0, lsr #23      // r2 = sign<<8 + exponent
  sub     r3, r2, r1, lsr #23  // shift = 0..254 (sign bits cancel)

  // Shift the mantissas up to the top of the words, and OR in the leading 1
  // for each.
  mov     r12, #1 << 31
  orr     r0, r12, r0, lsl #8
  orr     r1, r12, r1, lsl #8

LOCAL_LABEL(sub_dosub):
  // Here we perform the actual subtraction. We either fell through from the
  // code above, or jumped back to here after handling an input denormal.
  //
  // We get here with:
  //   Operands known to be numeric rather than zero/infinity/NaN;
  //   r0 = mantissa of larger operand (in high 24 bits);
  //   r1 = mantissa of smaller operand (in high 24 bits);
  //   r2 = result sign/exponent (in low 9 bits)
  //   r3 = exponent difference.
  //
  // Begin calculating the output mantissa by shifting y's mantissa right and
  // subtracting. This may leave the mantissa too large by one, if the bits
  // shifted out of y are nonzero. We correct this during rounding if
  // necessary.
#if !__thumb__
  subs    r12, r0, r1, lsr r3    // MI if high bit set
#else
  // Thumb can't fold a register-controlled shift into a sub, so we must use
  // two separate instructions.
  lsr     r12, r1, r3
  subs    r12, r0, r12
#endif

  // This may have cleared the high bit of the output mantissa, in which case
  // we must renormalise. Our strategy is to split into three code paths, on
  // two of which an awkward case is known not to arise:
  //  * no need to renormalise at all => underflow can't happen
  //  * shift up by exactly 1 bit
  //  * shift up by more than 1 bit => rounding can't happen (result is exact)
  //
  // First branch out of line for the first case, which we can detect because
  // the N flag tells us whether the top mantissa bit is still set.
  bmi     LOCAL_LABEL(sub_renorm_0)

  // Now we know we're renormalising by at least one bit, which also means
  // underflow is a risk.
  //
  // If we're shifting by only one bit, then underflow can only occur if the
  // exponent was originally 1. So test both those conditions together, and if
  // the shift is only one bit _and_ the exponent is > 1, we know we can
  // renormalise by one bit and not worry about underflow.
  tst     r2, #254  // test all but low bit of exponent; also clears N
#if !__thumb__
  movsne  r0, r12, lsl #1  // set N if non-underflowing _and_ top bit now set
#else
  // In Thumb, there's no advantage in combining the two tests, since the IT
  // between them costs a cycle. Do the explicit branch now to fsub_underflow
  // (because now we _know_ we have underflow).
  beq     LOCAL_LABEL(sub_underflow)
  // And then unconditionally do the shift.
  movs    r0, r12, lsl #1  // check whether 2nd bit is cleared (PL)
#endif
  // After all that, N is clear if we still haven't set the top mantissa bit,
  // either because we shifted up by a bit and it didn't help, or (in Arm state
  // only) because we detected underflow and didn't do the shift at all.
  //
  // The case of 'haven't yet done the shift' is reliably indicated by the Z
  // flag being set, because if we did do the shift, it will always have
  // cleared Z.
  bpl     LOCAL_LABEL(sub_renorm_orunder)

  // If we get here, we've renormalised by one bit (and have already shifted
  // the mantissa up), and we also know there's no underflow.
  //
  // Recombine the sign+exponent with the fraction. We must also decrement the
  // exponent, to account for the one-bit renormalisation. We do that by using
  // ASR to shift the mantissa right: its top bit is currently set, so the ASR
  // effectively puts -1 in the bits that are being added to the exponent.
  movs    r0, r0, asr #8  // also sets C if we need to round up
  adc     r0, r0, r2, lsl #23 // recombine, and also do basic rounding

  // If C was not set, then we've rounded down. Therefore, no need to round to
  // even, and also, no need to compensate for having shifted nonzero bits out
  // of the subtrahend. We can just return.
  bxcc    lr

  // If any bit shifted out of the 32-bit output mantissa is nonzero, then we
  // can also return, because we know we're rounding _up_ (and not to even),
  // and again, bits shifted out of the subtrahend don't matter because their
  // combined loss can't exceed the gain from one of these guard bits.
  tst     r12, #0x3F
  bxne    lr

  // Otherwise, we must do the full check for round to even.
  b       LOCAL_LABEL(sub_roundeven)

LOCAL_LABEL(sub_renorm_0):
  // We come here if no renormalisation is necessary, and therefore also no
  // underflow can happen.
  //
  // Since the leading bit is set, we need to decrement the exponent, to
  // account for the leading bit adding 1 to it when we recombine.
  movs    r0, r12, lsr #8  // also sets C if we need to round up
  sub     r2, r2, #1       // adjust exponent
  adc     r0, r0, r2, lsl #23 // recombine, and also do basic rounding

  // As in the 1-bit case above, if we didn't round up just now then we're
  // done, and if any bit shifted out of r12 just now was nonzero then we're
  // also done.
  bxcc    lr               // rounding down, done
  tst     r12, #0x7F
  bxne    lr               // nonzero guard bit, rounding up, done

  // Otherwise, fall through to the full check for round to even.
LOCAL_LABEL(sub_roundeven):
  // Same round-to-even check as in the fadd cases: find all the bits we
  // shifted out of y's mantissa and see if any are zero.
  rsb     r3, r3, #32
  lsls    r1, r1, r3       // set Z if we're rounding to even

  // Unlike the addition case, if we aren't rounding to even then the result is
  // currently too _big_: the top 32 bits of the output mantissa looked as if
  // they were on a rounding boundary, but those nonzero bits shifted off the
  // bottom of the mantissa make the true value slightly smaller than it
  // looked, so in fact we're just _below_ a rounding boundary. But we've
  // already rounded it up! So in the non-RTE case we must decrement the
  // output value.
  subne   r0, r0, #1       // no RTE, so undo round up
  biceq   r0, r0, #1       // yes RTE, so clear low bit of output
  bx      lr

LOCAL_LABEL(sub_renorm_orunder):
  // We come here if _either_ of these is true:
  //
  //  1. we've shifted the output mantissa left by one bit already but its top
  //     bit is still 0, so we must renormalise by more than 1 bit (and this
  //     may cause an underflow that we haven't detected yet)
  //
  //  2. (Arm only) we have detected an underflow already, not yet shifted the
  //     output mantissa at all, and haven't yet branched to fsub_underflow.

  // Get the output sign bit by itself in r3. This is needed by the code below,
  // and also used by fsub_underflow, so if we compute it before the (Arm-only)
  // branch to fsub_underflow then it doesn't have to be duplicated there.
  mov     r3, r2, lsr #8   // r3 now has just the output sign, in bit 0

#if !__thumb__
  // Arm state: we did a combined check for cases 1 and 2 above, so this is
  // where we separate them and go off to handle underflow in case 2. As stated
  // above, the Z flag indicates an already-detected underflow.
  beq     LOCAL_LABEL(sub_underflow)
#endif

  // Now we know that we must renormalise by at least 2 bits, which may also
  // give a denormal or zero result.
  //
  // This means no rounding can possibly be needed: if the subtraction cleared
  // the top two bits of the mantissa, it means we computed A-B and found it
  // was less than A/2, so B > A/2, so the exponent difference was at most 1.
  // Hence the result mantissa fits in 24 bits even before renormalisation, and
  // the top bit is clear, so it fits in 23 bits, i.e. it is exact.
  //
  // (That argument applies to the result before denormalisation. But any
  // subtraction delivering a denormal result must also be exact: the inputs to
  // subtraction are integer multiples of the smallest denormal, hence so is
  // the result.)

  // Start by shifting up by two bits (we already know the top 2 bits are
  // clear). In the process, test if the entire mantissa is actually zero.
  //
  // If the mantissa is zero, we can safely return +0. (In default IEEE
  // round-to-nearest mode, the only case of addition/subtraction that delivers
  // -0 is if you add two zeroes _both_ of which are -0, or the equivalent
  // subtraction. And those cases won't have come here, because they were
  // additions of like-signed inputs or subtraction of opposite-signed inputs,
  // so they go to fadd instead of fsub.)
  movs    r0, r0, lsr #2
  bxeq    lr               // result is zero, which r0 already contains

  // Determine how many more bits we need to shift the mantissa up, by counting
  // its leading zeroes. Adjust the exponent, and shift the mantissa into its
  // final position (assuming the output is still a normalised number).
  clz     r12, r0          // compute the shift / exponent adjustment
  sub     r2, r2, r12      // adjust exponent
  lsl     r0, r0, r12      // shift mantissa up to the top of the word
  lsr     r0, r0, #8       // and then down to its final position

  // Check for underflow. This occurs precisely when the adjustment to the
  // exponent in the bottom 8 bits of r2 carried into its sign bit (because at
  // the moment the value in r2 is one lower than the true output exponent, so
  // that adding the leading 1 bit in the mantissa will increment it back to
  // the correct value). So we can check the sign bit in r2 against the copy of
  // it we saved in r3 earlier. If no underflow, then we can just recombine the
  // sign and exponent with the mantissa (no rounding is needed on this branch)
  // and return.
  teq     r3, r2, lsr #8      // Exponent underflow?
  addeq   r0, r0, r2, lsl #23 // if so, trivially put the output back together
  bxeq    lr                  // and return

  // Now we _have_ underflowed, and the out-of-range exponent stored in the low
  // 8 bits of r2 tell us by how much: if it's -n, then we need to shift the
  // normalised mantissa down by n bits. So to make the output denormal, all we
  // have to do is to shift the mantissa down and recombine it with the
  // original sign in r3.
  //
  // Bit 8 of r2 contains a corrupted version of the sign bit, but we can
  // safely ignore that, because the semantics of AArch32 register-controlled
  // shift instructions are that only the low 8 bits of the shift-count
  // register are examined. So that sign bit is too high up to affect what
  // happens.

  rsb     r2, r2, #0           // r2 is now the shift count
LOCAL_LABEL(sub_do_underflow):             // we can also come here from below
  mov     r0, r0, lsr r2       // shift the mantissa down
  orr     r0, r0, r3, lsl #31  // put the sign back on
  bx      lr                   // and return

LOCAL_LABEL(sub_underflow):
  // We come here if we detected underflow in the 'renormalise by 1 bit' case.
  // So the input exponent must have been 1, and we shift the mantissa by only
  // one bit. The only question is whether we put the output sign on: if the
  // result is actually zero, we don't need to, because a subtraction giving a
  // zero output always gives +0 (as mentioned above).
  movs    r0, r12, lsr #8         // Denormalise and check if result is zero
  bxeq    lr                      // Return +0 if result is zero
#if __thumb__
  // Get the output sign in r3. In Arm this was already done just after start
  // of fsub_renorm_orunder, which all underflows went through. But in Thumb we
  // might have come straight here without setting up r3.
  mov     r3, r2, lsr #8
#endif
  orr     r0, r0, r3, lsl #31  // put the sign back on
  bx      lr                   // and return

LOCAL_LABEL(sub_uncommon):
  // We come here if the entry-point check says that at least one of x and y
  // has an uncommon (FF or 00) exponent. So we have at least one NaN,
  // infinity, denormal or zero, but we don't know which, or which operand it's
  // in. And we could have any combination of those types of input, in _both_
  // operands.

  // Detect FF exponents (NaNs or infinities) and branch again for those.
  mov     r12, #0xFF000000
  bics    r2, r12, r0, lsl #1
  bicsne  r2, r12, r1, lsl #1
  beq     LOCAL_LABEL(sub_naninf)

  // Now we know both inputs are finite, but there may be denormals or zeroes.
  // So it's safe to do the same sign check and cross-jump as we did on the
  // fast path.
  teq     r0, r1             // opposite signs?
  eormi   r1, r1, #1 << 31   // if so, negate the second operand
  bmi     LOCAL_LABEL(add_zerodenorm)    // and cross-jump to the fadd version of this code

LOCAL_LABEL(sub_zerodenorm):
  // Now we know x and y have the same sign, and at least one of them is zero
  // or denormal. If there aren't any zeroes, we'll end up rejoining the fast
  // path, so we must set up all the same registers, and do our checks for zero
  // in line with that.
  //
  // Start by exactly repeating the initial fast-path setup code: sort into
  // magnitude order, get the output sign+exponent and the exponent shift.
  subs    r2, r0, r1         // compare inputs, also keeping x-y
  eorlo   r2, r2, #1 << 31   // if misordered, flip high bit of difference
  sublo   r0, r0, r2         // and use that to swap and sign-flip
  addlo   r1, r1, r2         //   the two inputs
  mov     r2, r0, lsr #23      // r2 = sign<<8 + exponent
  sub     r3, r2, r1, lsr #23  // shift = 0..254 (sign bits cancel)

  // Shift y's mantissa up to the top of r1. We know y has exponent 0 (at least
  // one of the inputs does, and we've sorted them by now). So we definitely
  // don't need to set the leading bit on y's mantissa; also, if r1 becomes
  // zero, then we know we're subtracting 0 from x.
  movs    r1, r1, lsl #8
  beq     LOCAL_LABEL(sub_yzero)

  // Now we know there aren't any zeroes, and that y is a denormal. x might or
  // might not be a denormal, so we must check that and decide whether to set
  // its top mantissa bit.
  mov     r0, r0, lsl #8       // shift mantissa of x to the top of r0
  tst     r2, #255             // is x's exponent 0? If so, it's denormal
  orrne   r0, r0, #1 << 31     // if not, set leading bit of x,
  subne   r3, r3, #1           //   adjust exponent difference,

  b       LOCAL_LABEL(sub_dosub)

LOCAL_LABEL(sub_yzero):
  // Here, we know y = 0, so we're subtracting 0 from x. For most values of x,
  // we return x unchanged: subtracting 0 makes no difference. But if x is
  // _also_ 0 then we must return +0, rather than whatever x's sign of zero is.
  // (Because +0 is always the sign of zero you return when subtracting a
  // number from itself).
  movs    r12, r0, lsl #1      // test if x = 0 (bottom 31 bits all zero)
  moveq   r0, #0               // if so, replace x with +0
  bx      lr

LOCAL_LABEL(sub_naninf):
  // We come here if at least one input is a NaN or infinity. If either or both
  // inputs are NaN then we hand off to __fnan2 which will propagate a NaN from
  // the input.
  mov     r12, #0xFF000000
  cmp     r12, r0, lsl #1          // if (r0 << 1) > 0xFF000000, r0 is a NaN
  blo     SYMBOL_NAME(__compiler_rt_fnan2)
  cmp     r12, r1, lsl #1
  blo     SYMBOL_NAME(__compiler_rt_fnan2)

  // Otherwise, we have no NaNs and at least one infinity, so we're returning
  // either infinity, or NaN for an (inf-inf) subtraction. We can safely handle
  // all these cases by flipping the sign of y and going to fadd_inf.
  eor     r1, r1, #0x80000000
  b       LOCAL_LABEL(add_inf)

END_COMPILERRT_FUNCTION(__aeabi_fsub)

NO_EXEC_STACK_DIRECTIVE

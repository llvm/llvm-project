//===-- mulsf3.S - single-precision floating point multiplication ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements single-precision soft-float multiplication with the
// IEEE-754 default rounding (to nearest, ties to even), in optimized AArch32
// assembly language suitable to be built as either Arm or Thumb2.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"


  .syntax unified
  .text
  .p2align 2

DEFINE_AEABI_FUNCTION_ALIAS(__aeabi_fmul, __mulsf3)

DEFINE_COMPILERRT_FUNCTION(__mulsf3)

  // Check if either input exponent is 00 or FF (i.e. not a normalized number),
  // and if so, branch out of line. If we don't branch out of line, then we've
  // also extracted the exponents of the input values r0/r1 into bits 16..23 of
  // r2/r3. But if we do, then that hasn't necessarily been done (because the
  // second AND might have been skipped).
  MOV     r12, #0xFF0000
  ANDS    r2, r12, r0, LSR #7  // sets Z if exponent of x is 0
  ANDSNE  r3, r12, r1, LSR #7  // otherwise, sets Z if exponent of y is 0
  TEQNE   r2, r12              // otherwise, sets Z if exponent of x is FF
  TEQNE   r3, r12              // otherwise, sets Z if exponent of y is FF
  BEQ     LOCAL_LABEL(uncommon)        // branch out of line to handle inf/NaN/0/denorm

  // Calculate the sign of the result, and put it in an unused bit of r2.
  TEQ     r0, r1               // sets N to the XOR of x and y's sign bits
  ORRMI   r2, r2, #0x100       // if N set, set bit 8 of r2

  // Move the input mantissas to the high end of r0/r1, each with its leading
  // bit set explicitly, so that they're in the right form to be multiplied.
  MOV     r12, #0x80000000
  ORR     r0, r12, r0, LSL #8
  ORR     r1, r12, r1, LSL #8

  // Now we're ready to multiply mantissas. This is also the place we'll come
  // back to after decoding denormal inputs. The denormal decoding will also
  // have to set up the same register contents:
  //  - decoded fractions at the top of r0 and r1
  //  - exponents in r2 and r3, starting at bit 16
  //  - output sign in r2 bit 8
LOCAL_LABEL(mul):

  // Here we multiply the mantissas, and compute the output exponent by adding
  // the input exponents and rebiasing. These operations are interleaved to
  // use a delay slot.
  //
  // The exponent is rebiased by subtracting 0x80, rather than the 0x7F you'd
  // expect. That compensates for the leading bit of the mantissa overlapping
  // it, when we recombine the exponent and mantissa by addition.
  ADD     r2, r2, r3           // r2 has sum of exponents, freeing up r3
  UMULL   r1, r3, r0, r1       // r3:r1 has the double-width product
  SUB     r2, r2, #(0x80 << 16) // rebias the summed exponent

  // Compress the double-word product into just the high-order word r3, by
  // setting its bit 0 if any bit of the low-order word is nonzero. This
  // changes the represented value, but not by nearly enough to affect
  // rounding, because rounding only depends on the bit below the last output
  // bit, and the general question of whether _any_ nonzero bit exists below
  // that.
  CMP     r1, #0                // if low word of full product is nonzero
  ORRNE   r3, r3, #1            //   then set LSB of high word

  // The two inputs to UMULL had their high bits set, that is, were at least
  // 0x80000000. So the 64-bit product was at least 0x4000000000000000, i.e.
  // the high bit of the product could be at the top of the word or one bit
  // below. Check which, by experimentally shifting left, and then undoing it
  // via RRX if we turned out to have shifted off a 1 bit.
  LSLS    r3, r3, #1            // shift left, setting C to the bit shifted off
  RRXCS   r3, r3                // if that bit was 1, put it back again

  // That ensured the leading 1 bit of the product is now the top of r3, but
  // also, set C if the leading 1 was _already_ in the top bit. So now we know
  // whether to increment the exponent. The following instruction does the
  // conditional increment (because it's ADC), but also, copies the exponent
  // field from bit 16 of r2 into bit 0, so as to place it just below the
  // output sign bit.
  //
  // So, if the number hasn't overflowed or underflowed, the low 9 bits of r2
  // are exactly what we need to combine with the rounded mantissa. But the
  // full output exponent (with extra bits) is still available in the high half
  // of r2, so that we can check _whether_ we overflowed or underflowed.
  ADC     r2, r2, r2, ASR #16

  // Recombine the exponent and mantissa, doing most of the rounding as a side
  // effect: we shift the mantissa right so as to put the round bit into C, and
  // then we recombine with the exponent using ADC, to increment the mantissa
  // if C was set.
  MOVS    r12, r3, LSR #8
  ADC     r0, r12, r2, LSL #23

  // To complete the rounding, we must check for the round-to-even tiebreaking
  // case, by checking if we're in the exact halfway case, which occurs if and
  // only if we _did_ round up (we can tell this because C is still set from
  // the MOVS), and also, no bit of r3 is set _below_ the round bit.
  //
  // We combine this with an overflow check, so that C ends up set if anything
  // weird happened, and clear if we're completely finished and can return.
  //
  // The best instruction sequence for this part varies between Arm and Thumb.
#if !__thumb__
  // Arm state: if C was set then we check the low bits of r3, so that Z ends
  // up set if we need to round to even.
  //
  // (We rely here on Z reliably being clear to begin with, because shifting
  // down the output mantissa definitely gave a nonzero output. Also, the TST
  // doesn't change C, so if Z does end up set, then C was also set.)
  //
  // Then, if we're not rounding to even, we do a CMP which sets C if there's
  // been an overflow or an underflow. An overflow could occur for an output
  // exponent as low as 0xFC, because we might increment the exponent by 1 when
  // renormalizing, by another when recombining with the mantissa, and by one
  // more if rounding up causes a carry off the top of the mantissa. An
  // underflow occurs only if the output exponent is negative (because it's
  // offset by 1, so an exponent of 0 will be incremented to 1), in which case
  // the top 8 bits of r2 will all be set. Therefore, an unsigned comparison to
  // see if r2 > 0xFC0000 will catch all overflow and underflow cases. It also
  // catches a few very large cases that _don't_ quite overflow (exponents of
  // 0xFC and above that don't get maximally unlucky); those will also be
  // handled by the slow path.
  TSTCS   r3, #0x7F
  CMPNE   r2, #0xFC0000
#else
  // In Thumb, switching between different conditions has a higher cost due to
  // the (implicit in this code) IT instructions, so we prefer a strategy that
  // uses CC and CS conditions throughout, at the cost of requiring some extra
  // cleanup instructions on the slow path.
  //
  // If C is set (and hence round-to-even is a possibility), the basic idea is
  // to shift the full result word (r3) left by 25, leaving only its bottom 7
  // bits, which are now the top 7 bits; then we want to set C iff these are 0.
  //
  // The "CMP x,y" instruction sets C if y > x (as unsigned integers). So this
  // could be done in one instruction if only we had a register to use as x,
  // which has 0 in the top 7 bits and at least one nonzero. Then we could
  // compare that against the shifted-up value of r3, setting C precisely if
  // the top 7 bits of y are greater than 0. And happily, we _do_ have such a
  // register! r12 contains the shifted-down mantissa, which is guaranteed to
  // have a 1 in bit 23, and 0 above that.
  //
  // The shift of r3 happens only in the second operand of the compare, so we
  // don't lose the original value of r3 in this process.
  //
  // The check for over/underflow is exactly as in the Arm branch above, except
  // based on a different condition.
  CMPCS   r12, r3, LSL #25  // now C is set iff we're rounding to even
  CMPCC   r2, #0xFC0000     // and now it's also set if we've over/underflowed
#endif

  // That's all the checks for difficult cases done. If C is clear, we can
  // return.
  BXCC    lr

  // Now the slower path begins. We have to recover enough information to
  // handle all of round-to-even, overflow and underflow.
  //
  // Round to even is the most likely of these, so we detect it first and
  // handle it as fast as possible.

#if __thumb__
  // First, Thumb-specific compensation code. The Arm branch of the #if above
  // will have set Z=0 to indicate round to even, but the Thumb branch didn't
  // leave any unambiguous indicator of RTE, so we must retest by checking all
  // the bits shifted off the bottom of the mantissa to see if they're exactly
  // the half-way value.
  LSL     r12, r3, #24           // r12 = round bit and everything below
  CMP     r12, #0x80000000       // set Z if that is exactly 0x80000000
#endif

  // Now Z is clear iff we have already rounded up and now must replace that
  // with rounding to even, which is done by just clearing the low bit of the
  // mantissa.
  BICEQ   r0, r0, #1

  // Redo the over/underflow check (the same way as in both branches above),
  // and if it doesn't report a danger, we can return the rounded-to-even
  // answer.
  CMP     r2, #0xFC0000         // check for over/underflow
  BXCC    lr                    // and return if none.

  // Now we only have overflow and underflow left to handle. First, find out
  // which we're looking at. This is easy by testing the top bit of r2, but
  // even easier by using the fact that the possible positive and negative
  // values of r2 are widely enough separated that the 0xFC0000 subtracted by
  // the CMP above won't have made any difference. So the N flag output from
  // that comparison _already_ tells us which condition we have: if N is set we
  // have underflow, and if N is clear, overflow.
  BPL     LOCAL_LABEL(overflow)

  // Here we're handling underflow.

  // Add the IEEE 754:1985 exponent bias which funder will expect. This also
  // brings the exponent back into a range where it can't possibly have carried
  // into the sign bit, so the output sign will now be right.
  ADD     r0, r0, #(0xC0 << 23)

  // Determine whether we rounded up, down or not at all.
  LSLS    r2, r3, #1              // input mantissa, without its leading 1
  SUBS    r1, r2, r0, LSL #9      // subtract the output mantissa (likewise)

  // And let funder handle the rest.
  B     SYMBOL_NAME(__compiler_rt_funder)

LOCAL_LABEL(overflow):
  // We come here to handle overflow, but it's not guaranteed that an overflow
  // has actually happened: our check on the fast path erred on the side of
  // caution, by catching any output exponent that _could_ cause an overflow.
  // So first check whether this really is an overflow, by extracting the
  // output exponent. Exponent 0xFF, or anything that wrapped round to having
  // the high bit clear, are overflows; 0xFE down to 0xFC are not overflows.
  //
  // The value in r0 is correct to return, if there's no overflow.
  ADD     r12, r0, #(1 << 23)     // add 1 to the exponent so 0xFF wraps to 0
  MOVS    r12, r12, LSL #1        // test the top bit of the modified value
  BXMI    lr                      // if top bit is still 1, not an overflow

  // This is an overflow, so we need to replace it with an appropriately signed
  // infinity. First we correct the sign by applying a downward bias to the
  // exponent (the one suggested in IEEE 754:1985, which was chosen to bring
  // all possible overflowed results back into range).
  SUBS    r0, r0, #(0xC0 << 23)

  // Now the sign bit of r0 is correct. Replace everything else with the
  // encoding of an infinity.
  MOV     r1, #0xFF
  AND     r0, r0, #0x80000000
  ORR     r0, r0, r1, LSL #23
  BX      lr

LOCAL_LABEL(uncommon):
  // Handle zeros, denorms, infinities and NaNs. We arrive here knowing that
  // we've at least done the first _two_ instructions from the entry point,
  // even if all the rest were skipped. So r2 contains the sign and exponent of
  // x in bits 16..23, and r12 = 0xFF << 16.
  //
  // So, first repeat some instructions from the prologue, which were either
  // conditionally skipped in the sequence leading to the branch, or skipped
  // because they happened after the branch.
  AND     r3, r12, r1, LSR #7  // get exponent of y in r3 bits 16..23
  TEQ     r0, r1               // calculate the sign of the result
  ORRMI   r2, r2, #0x100       // and put it in bit 8 of r2 as before

  // Check for infinities and NaNs, by testing each of r2,r3 to see if it's at
  // least 0xFF0000 (hence the exponent field is equal to 0xFF).
  CMP     r2, r12
  CMPLO   r3, r12
  BHS     LOCAL_LABEL(inf_NaN)

  // If we didn't take that branch, then we have only finite numbers, but at
  // least one is denormal or zero. A zero makes the result easy (and also is a
  // more likely input than a denormal), so check those first, as fast as
  // possible.
  MOVS    r12, r0, LSL #1          // Z set if x == 0
  MOVSNE  r12, r1, LSL #1          // now Z set if either input is 0
  MOVEQ   r0, r2, LSL #23          // in either case, make 0 of the output sign
  BXEQ    lr                       // and return it

  // Now we know we only have denormals to deal with. Call fnorm2 to sort
  // them out, and rejoin the main code path above.
  AND     r12, r2, #0x100          // save the result sign from r2
  LSR     r2, #16                  // shift extracted exponents down to bit 0
  LSR     r3, #16                  // where fnorm2 will expect them
  PUSH    {r0, r1, r2, r3, r12, lr}
  MOV     r0, sp                   // tell fnorm2 where to find its data
  BL      SYMBOL_NAME(__compiler_rt_fnorm2)
  POP     {r0, r1, r2, r3, r12, lr}
  LSL     r3, #16                  // shift exponents back up to bit 16
  ORR     r2, r12, r2, LSL #16     // and put the result sign back in r2
  B       LOCAL_LABEL(mul)

LOCAL_LABEL(inf_NaN):
  // We come here if at least one input is a NaN or infinity. If either or both
  // inputs are NaN then we hand off to fnan2 which will propagate a NaN from
  // the input; otherwise any multiplication involving infinity returns
  // infinity, unless it's infinity * 0 which is an invalid operation and
  // returns NaN again.
  MOV     r12, #0xFF000000
  CMP     r12, r0, LSL #1          // if (r0 << 1) > 0xFF000000, r0 is a NaN
  BLO     SYMBOL_NAME(__compiler_rt_fnan2)
  CMP     r12, r1, LSL #1
  BLO     SYMBOL_NAME(__compiler_rt_fnan2)

  // NaNs are dealt with, so now we have at least one infinity. Check if the
  // other operand is 0. This is conveniently done by XORing the two: because
  // we know that the low 31 bits of one operand are exactly 0x7F800000, we can
  // test if the low 31 bits of the other one are all 0 by checking whether the
  // low 31 bits of (x XOR y) equal 0x7F800000.
  EOR     r3, r0, r1
  CMP     r12, r3, LSL #1          // if inf * 0, this sets Z
  LSR     r0, r12, #1              // set up return value of +infinity
  ORRNE   r0, r0, r2, LSL #23      // if not inf * 0, put on the output sign
  ORREQ   r0, r0, #0x400000        // otherwise, set the 'quiet NaN' bit
  BX      lr                       // and return

END_COMPILERRT_FUNCTION(__mulsf3)

NO_EXEC_STACK_DIRECTIVE

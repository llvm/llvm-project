//===-- muldf3.S - double-precision floating point multiplication ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __muldf3 function (double precision floating point
// multiplication), with the IEEE-754 default rounding (to nearest, ties to
// even), for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"
#include "endian.h"

  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__muldf3)
  push {r4, lr}
  VMOV_FROM_DOUBLE(r0, r1, d0)
  VMOV_FROM_DOUBLE(r2, r3, d1)
  bl __aeabi_dmul
  VMOV_TO_DOUBLE(d0, r0, r1)
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__muldf3, __aeabi_dmul)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_dmul)

  push    {r4,r5,r6,lr}

  // Check if either input exponent is 000 or 7FF (i.e. not a normalized
  // number), and if so, branch out of line. If we don't branch out of line,
  // then we've also extracted the exponents of the input values x and y into
  // bits 16..26 of r14 and r5 respectively. But if we do, then that hasn't
  // necessarily been done (because the second AND might have been skipped).
  ldr     r12, =0x07FF0000
  ands    r14, r12, xh, lsr #4 // sets Z if exponent of x is 0
  andsne  r5, r12, yh, lsr #4  // otherwise, sets Z if exponent of y is 0
  teqne   r14, r12             // otherwise, sets Z if exponent of x is 7FF
  teqne   r5, r12              // otherwise, sets Z if exponent of y is 7FF
  beq     LOCAL_LABEL(uncommon)        // branch out of line to handle inf/NaN/0/denorm

  // Calculate the sign of the result, and put it in an unused bit of r14.
  eor     r4, xh, yh           // XOR the input signs to get the result sign
  orr     r14, r14, r4, lsr #31 // save it in the low bit of r14

  // Clear the exponent and sign bits from the top word of each mantissa, and
  // set the leading mantissa bit in each one, so that they're in the right
  // form to be multiplied.
  bic     xh, xh, r12, lsl #5     // r12 = 0x07FF0000, so r12 << 5 = 0xFF800000
  bic     yh, yh, r12, lsl #5
  orr     xh, xh, #1<<20
  orr     yh, yh, #1<<20

  // Now we're ready to multiply mantissas. This is also the place we'll come
  // back to after decoding denormal inputs. The denormal decoding will also
  // have to set up the same register contents:
  //  - fractions in xh/xl and yh/yl, with leading bits at bit 20 of xh/yh
  //  - exponents in r14 and r5, starting at bit 16
  //  - output sign in r14 bit 0
LOCAL_LABEL(mul):

  // Multiply the two mantissas as if they were full 64-bit words, delivering a
  // 128-bit output in four registers. We provide three different ways to do
  // this, using different instructions.
  //
  // Interleaved with the multiplication code, we also compute the output
  // exponent by adding the input exponents and rebiasing. This takes two
  // instructions. We schedule each one after a multiplication, to use a delay
  // slot from the multiplication on CPUs where there is one.
  //
  // We add r5 to r14, so that the output exponent is in the top half of r14,
  // and r5 is freed up to be used in the multiplication.
  //
  // We rebias the exponent by subtracting 0x400, which is correct for one of
  // the two places where the leading bit of the product could end up, and will
  // need correcting by one in the other case.
  //
  // Exit conditions from the three-way #if:
  //
  // r4:r5:r6 are the top 96 bits of the 128-bit product, with the leading bit
  // at either bit 8 or bit 9 of r4. The low bit of r6 is forced to 1 if any of
  // the low 32 bits of the 128-bit product were set.
  //
  // The output sign is still in the low bit of r14; the top half contains the
  // preliminary output exponent (yet to be adjusted depending on where the
  // high bit of the product ended up).

#if __ARM_FEATURE_DSP
  // The UMAAL instruction, which computes a 64-bit product and adds two
  // separate 32-bit values to it, makes this easy.
  umull   r6, r4, xh, yl
  add     r14, r14, r5             // add exponents, freeing up r5
  umull   r12, r5, xl, yl
  sub     r14, r14, #0x4000000     // initial rebiasing of exponent
  umaal   r6, r5, xl, yh
  umaal   r5, r4, xh, yh
#elif ARM_FP_DMUL_USE_UMLAL
  // The UMLAL instruction computes a 64-bit product and adds a 64-bit value to
  // it. But it doesn't write to the carry flag, so you can't tell if the
  // addition wrapped. Therefore you have to use it in a way that means the
  // addition never wraps. Here we do three of the four multiplications (xl*yl,
  // xl*yh, xh*yh) in a chain, using UMLAL for the top two, in each case with
  // the 64-bit accumulator consisting of the top half of the previous
  // multiplication, and a high word set to zero before the UMLAL instruction.
  //
  // On Cortex-M3, this is not a win over just using UMULL and doing the
  // additions by hand, because UMLAL takes two cycles longer than UMULL, and
  // it also costs a cycle to initialise each of the two high accumulator words
  // to zero. If the high word of the addend were not zero then those two
  // cycles would be doing something useful, but as it is, they're wasted time.
  //
  // CPUs later than Cortex-M3 - in particular, Cortex-M4 - will do both UMLAL
  // and UMULL much faster, so that this code is a win over the plain UMULL
  // code below. But those CPUs typically have UMAAL anyway and will use the
  // even faster version of the code above. So this code is provided in case
  // it's useful, but won't be enabled unless you manually #define
  // ARM_FP_DMUL_USE_UMLAL.
  umull   r12, r6, xl, yl
  add     r14, r14, r5             // add exponents, freeing up r5
  movs    r5, #0
  umlal   r6, r5, xl, yh
  movs    r4, #0
  umlal   r5, r4, xh, yh
  sub     r14, r14, #0x4000000     // initial rebiasing of exponent
  umull   xl, yh, xh, yl
  adds    r6, r6, xl
  adcs    r5, r5, yh
  adc     r4, r4, #0
#else
  // Simplest approach, using plain UMULL to compute each 64-bit product, and
  // separate ADD and ADC instructions to do the additions. On Cortex-M3 this
  // wins over the UMLAL approach: it's one instruction longer, but three
  // cycles quicker, since each use of UMLAL in the above version costs 2
  // cycles.
  umull   r4, r12, xh, yl
  add     r14, r14, r5             // add exponents, freeing up r5
  umull   r6, r5, xl, yh
  sub     r14, r14, #0x4000000     // initial rebiasing of exponent
  adds    r6, r6, r4
  adcs    r5, r5, r12              // carry from here is used below

  umull   r4, r12, xh, yh          // r12:r4 is top part
  adc     yh, r12, #0              // get carry from above addition
  umull   r12, xh, xl, yl          // xh:r12 is bottom part

  adds    r6, r6, xh
  adcs    r5, r5, r4
  adcs    r4, yh, #0
#endif

  // Now the full 128-bit product of the two mantissas occupies the four
  // registers r4,r5,r6,r12 (in order from MSW to LSW). Since each input
  // mantissa was in the range [2^52,2^53), the product is in the range
  // [2^104,2^106), which means that the lowest-order word r12 is a long way
  // below the round bit, so that it can only affect cases so close to a
  // rounding boundary that you need to know if it's nonzero to tell whether
  // you're rounding to even. Start by freeing up that register, ensuring the
  // low bit of r6 is set if anything in r12 was nonzero.
  tst     r12, r12
  orrne   r6, r6, #1

  // Now we can regard the result as a 96-bit value in r4,r5,r6, with its
  // leading bit in either bit 8 or 9 of r4. To move that bit up to its final
  // position in bit 20, we must shift the whole thing left by either 11 or 12
  // bits. Find out which.
  tst     r4, #0x200               // is bit 9 set?
  bne     LOCAL_LABEL(shift11)             // if so, only shift by 11 bits

  // In this branch, we're shifting left by 12 bits. Put the shifted result
  // back into the output registers xh,xl, and the bits lower than the bottom
  // mantissa bit into r4.
  lsls    xh, r4, #12              // shift each input reg left 12
  lsls    xl, r5, #12
  lsls    r4, r6, #12
  orr     xh, xh, r5, lsr #20      // and the top two right by 32-12
  orr     xl, xl, r6, lsr #20

  b       LOCAL_LABEL(shifted)

LOCAL_LABEL(shift11):
  // In this branch, we're shifting left by 11 bits instead of 12, and we must
  // adjust the exponent by 1 to compensate.
  lsls    xh, r4, #11              // shift each input reg left 11
  lsls    xl, r5, #11
  lsls    r4, r6, #11
  orr     xh, xh, r5, lsr #21      // and the top two right by 32-11
  orr     xl, xl, r6, lsr #21
  add     r14, r14, #0x10000       // adjust the exponent

LOCAL_LABEL(shifted):
  // We've reconverged after shifting the mantissa, so that now the leading 1
  // bit of the mantissa is in bit 20 of xh, and r4 contains the bits lower
  // than the bottom of xl.

  // Recombine the sign and exponent into the high bits of xh. If the exponent
  // is over- or underflowed, this may not give a valid FP result, but because
  // everything is put on by addition, it will be right "mod 2^64" so that we
  // can bias the exponent back into range for underflow handling and that will
  // recover the right sign.
  //
  // r14 still has the output sign in its low bit. To extract just the exponent
  // for adding to xh, we could use BIC to clear that bit, or shift the value
  // right. We do the latter, which saves a copy of the pre-rounding exponent
  // in yl, to use later for overflow detection. The shift is ASR, so that if
  // the exponent is negative due to underflow, it stays negative.
  asr     yl, r14, #16             // isolate the exponent
  add     xh, xh, yl, lsl #20      // shift it back up to add to xh
  add     xh, xh, r14, lsl #31     // then add the sign

  // If we have to handle an underflow, we'll need enough information to
  // reconstruct the rounding direction. Our strategy is
  //
  //  - save the LSW of the output before rounding: if that differs from the
  //    LSW after rounding then we rounded up
  //  - save the round word r4: if that is zero then we didn't round at all.
  //
  // We're going to branch past the rounding code for a quicker exit in the
  // case where we're exact. In that case we don't need to save the output LSW
  // at all, because the zero round word will override whatever it would have
  // been anyway.
  movs    r6, r4                   // unconditionally save round word
  beq     LOCAL_LABEL(rounded)             // branch past rounding code if exact
  mov     r5, xl                   // and if not, save output LSW too

  // Rounding: we shift r4 left to put the round bit into the carry flag so
  // that ADCS+ADC will conditionally increment the mantissa. But before we do
  // the additions, we also check the Z flag, which tells us whether the
  // remaining 31 bits are all zero. If so, we're either in the round-to-even
  // (RTE) halfway case, or the exact case - but the exact case never came
  // through this code at all, so it must be RTE.
  //
  // If those 31 bits _aren't_ all zero, we clear the top bit of r4, leaving it
  // set only in the round-to-even case. Then (r4 >> 31) can be used to clear
  // the low bit to perform RTE.
  lsls    r12, r4, #1              // test round word
  bicne   r4, r4, #0x80000000      // make top bit of r4 into the RTE bit
  adcs    xl, xl, #0               // conditionally increment the mantissa
  adc     xh, xh, #0               // ... and carry into its high word
  bic     xl, xl, r4, lsr #31      // round to even if r4[31] != 0

LOCAL_LABEL(rounded):
  // Now we've rounded the output. The last thing we must do is check for
  // overflow and underflow: if neither has happened, we can return.
  //
  // yl contains the pre-rounding output exponent minus 1 (so that the leading
  // mantissa bit incremented it to the right output value). If this is in the
  // range [0,0x7fd] then the leading bit would have incremented it to
  // [1,0x7fe], which are non-overflowed output exponents. So an unsigned check
  // if yl >= 0x7fe detects both overflow and underflow at once.
  movw    r12, #0x7FE
  cmp     yl, r12
  poplo   {r4,r5,r6,pc}

  // We have either an underflow or an overflow. We can tell which it is by
  // doing a _signed_ comparison of yl with the same value again - and since we
  // only just did the CMP instruction, we can reuse the same flags.
  bge     LOCAL_LABEL(overflow)

  // Now we're dealing with an underflow. Set r2 to the rounding direction, by
  // first checking xl against r5 (where we saved its pre-rounding value) to
  // see if we rounded up or down, and then overriding that by checking r6
  // (where we saved the round word) to see if we didn't round at all. In the
  // latter case the comparison against r5 will deliver nonsense, but then we
  // overwrite it, so it doesn't matter.
  cmp     xl, r5                   // did we modify the LSW, i.e. round up?
  movne   r2, #-1                  // if so, the true value is a bit smaller
  moveq   r2, #+1                  // else it's a bit bigger
  cmp     r6, #0                   // except maybe we didn't round at all
  moveq   r2, #0                   // in which case the true value is exact.

  // Add the IEEE 754 exponent bias, and tail-call __dunder to handle the rest
  // of the job.
  add     xh, xh, #0x60000000
  pop     {r4,r5,r6,lr}
  b       SYMBOL_NAME(__compiler_rt_dunder)

LOCAL_LABEL(overflow):
  // Here, we overflowed, so we must return an infinity of the correct sign.
  // Rebias the exponent, which corrects the sign bit.
  sub     xh, xh, #0x60000000

  // And pop our scratch registers before falling through into dmul_retinf.
  pop     {r4,r5,r6,lr}

LOCAL_LABEL(retinf):
  // This is entered from the overflow handler and also from cases with
  // infinite inputs. It constructs an infinity, with sign bit equal to the
  // high bit of xh.
  //
  // On entry to here, we expect not to have a stack frame any more, because
  // one of our callers will have popped it already in order to conditionally
  // tailcall __dnan2.
  mov     xl, #0                   // clear low word
  mvn     xh, xh, lsr #31          // shift xh[31] down to bit 0, inverted
  mvn     xh, xh, lsl #11          // uninvert, and put exponent 0x7ff below it
  lsl     xh, xh, #20              // shift back up to the top
  bx      lr

LOCAL_LABEL(uncommon):
  // We come here from the entry point, if any input had exponent 0 or 0x7ff.
  // First we must repeat the instruction from the entry point that sets up r5
  // with the exponent of y, this time unconditionally, so we know we have both
  // exponents in the top halves of r14 and r5.
  and     r5, r12, yh, lsr #4

  // Check if either exponent is 0x7ff, by comparing against the value left in
  // r12 by the entry point. If so, branch away to handle NaNs and infinities.
  teq     r14, r12
  teqne   r5, r12
  beq     LOCAL_LABEL(naninf)

  // If we didn't branch, we're dealing with finite numbers, including a zero
  // or a denormal or both.
  //
  // First save the output sign.
  eor     r6, xh, yh

  // Handle zeroes first, because if there's a zero we don't have to worry
  // about denormals at all.
  orrs    r4, xl, xh, lsl #1      // is x zero?
  orrsne  r4, yl, yh, lsl #1      // or is y zero?
  beq     LOCAL_LABEL(retzero)            // Return zero if so

  // Otherwise, delegate to __dnorm2 to handle denormals, converting them into
  // a normalised mantissa and an out-of-range exponent. __dnorm2 expects the
  // exponents at the bottom of their words instead of half way up, so shift
  // down first, and back up again afterwards.
  //
  // This call clobbers r12, because we didn't bother to save it on the stack.
  // That's fine, because we don't need the constant in it any more. When we go
  // back to dmul_mul, that will use it as a scratch register.
  lsr     r4, r14, #16
  lsr     r5, r5, #16
  push    {r0, r1, r2, r3, r4, r5} // create a 'struct dnorm2' on the stack
  mov     r0, sp                   // pass it by address
  bl      SYMBOL_NAME(__compiler_rt_dnorm2)
  pop     {r0, r1, r2, r3, r4, r5}
  lsl     r14, r4, #16
  lsls    r5, r5, #16

  // Put the output sign at the bottom of r14, the same place the fast path
  // would have left it. Then rejoin the fast path.
  orr     r14, r14, r6, lsr #31
  b       LOCAL_LABEL(mul)

LOCAL_LABEL(retzero):
  // Return an exact zero, with sign bit from the high bit of r6.
  mov     xl, #0                  // low word is 0
  ands    xh, r6, #0x80000000     // high word is 0 except for the sign
  pop     {r4,r5,r6,pc}

LOCAL_LABEL(naninf):
  // We come here knowing that at least one operand is either NaN or infinity.
  // If there's a NaN, we can tailcall __dnan2 to do the right thing. Pop our
  // stacked registers first: we won't need that much spare space any more, and
  // it makes the tailcall easier if we've already done it.
  pop     {r4,r5,r6,lr}

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
  // either operand is zero then we have inf * 0 = invalid operation and must
  // return a NaN.
  orrs    r12, xl, xh, lsl #1     // are all bits of x zero except the sign?
  beq     LOCAL_LABEL(retnan)             // if so, x == 0, so y == inf
  orrs    r12, yl, yh, lsl #1     // same check the other way round
  beq     LOCAL_LABEL(retnan)

  // If we have an infinity and no NaN, then we just return an infinity of the
  // correct sign.
  eor     xh, xh, yh
  b       LOCAL_LABEL(retinf)

LOCAL_LABEL(retnan):
  // Return the default NaN, in the case where the inputs were 0 and infinity.
  movw    xh, 0x7ff8
  lsls    xh, xh, #16
  mov     xl, #0
  bx      lr

END_COMPILERRT_FUNCTION(__aeabi_dmul)

NO_EXEC_STACK_DIRECTIVE

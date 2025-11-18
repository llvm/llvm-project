//===-- mulsf3.S - single-precision floating point multiplication ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements single-precision soft-float multiplication with the
// IEEE-754 default rounding (to nearest, ties to even), in optimized Thumb1
// assembly language.
//
//===----------------------------------------------------------------------===//

#include "../../assembly.h"

  .syntax unified
  .text
  .thumb
  .p2align 2

DEFINE_AEABI_FUNCTION_ALIAS(__aeabi_fmul, __mulsf3)

DEFINE_COMPILERRT_THUMB_FUNCTION(__mulsf3)
  push {r4,r5,r6,lr}

  // Get exponents of the inputs, and check for uncommon values. In the process
  // of this we also compute the sign, because it's marginally quicker that
  // way.
  lsls    r2, r0, #1
  adcs    r4, r4, r4    // set r4[0] to sign bit of x
  lsls    r3, r1, #1
  adcs    r4, r4, r3    // set r4[0] to the output sign
  lsrs    r2, r2, #24
  beq     LOCAL_LABEL(zerodenorm0)   // still do the next LSRS
  lsrs    r3, r3, #24
  beq     LOCAL_LABEL(zerodenorm)
  cmp     r2, #255
  beq     LOCAL_LABEL(naninf)
  cmp     r3, #255
  beq     LOCAL_LABEL(naninf)
  // Compute the output exponent. We'll be generating our product _without_ the
  // leading bit, so we subtract 0x7f rather than 0x80.
  adds    r2, r2, r3
  subs    r2, r2, #0x7f
  // Blank off everything above the mantissas.
  lsls    r0, r0, #9
  lsls    r1, r1, #9
LOCAL_LABEL(normalised): // we may come back here from zerodenorm
  lsrs    r0, r0, #9
  lsrs    r1, r1, #9
  // Multiply. r0 and r1 are the mantissas of the inputs but without their
  // leading bits, so the product we want in principle is P=(r0+2^23)(r1+2^23).
  // P is at most (2^24-1)^2 < 2^48, so it fits in a word and a half.
  //
  // The technique below will actually compute P - 2^46, by not adding on the
  // term where the two 2^23 are multiplied. The 48-bit result will be
  // delivered in two output registers, one containing its bottom 32 bits and
  // the other containing the top 32, so they overlap in the middle 16 bits.
  // This is done using only two multiply instructions and some bookkeeping.
  //
  // In the comments I'll write X and Y for the original input mantissas (again
  // without their leading bits). I'll also decompose them as X = xh + xl and
  // Y = yh + yl, where xl and yl are in the range 0..2^8-1 and xh,yh are
  // multiples of 2^8.
  adds    r5, r0, r1
  lsls    r5, r5, #7    // r5 = (X+Y) << 7
  movs    r6, r0
  muls    r6, r1, r6    // r6 is congruent mod 2^32 to X*Y
  lsrs    r0, r0, #8
  lsrs    r1, r1, #8
  muls    r0, r1, r0
  lsls    r1, r0, #16   // r1 is congruent mod 2^32 to xh*yh
  subs    r3, r6, r1    // now r3 is congruent mod 2^32 to
                        //   (X*Y) - (xh*yh) = xh*yl + xl*yh + xl*yl
                        //   and hence, since that is at most 0xfeff0001,
                        //   is _exactly_ equal to that
  adds    r0, r0, r5    // r0 is now (xh*yh + (X+Y)<<23) >> 16
  lsrs    r1, r3, #16   // r1 is the top 16 bits of r3, i.e.
                        //   (xh*yl + xl*yh + xl*yl) >> 16
  adds    r3, r0, r1    // now r3 equals
                        //   (xh*yh + xh*yl + xl*yh + xl*yl + (X+Y)<<23) >> 16
                        //   i.e. (X*Y + (X+Y)<<23) >> 16,
                        //   i.e. (the right answer) >> 16.
                        // Meanwhile, r6 is exactly the bottom 32 bits of the
                        // right answer.
  // Renormalise if necessary.
  lsrs    r1, r3, #30
  beq     LOCAL_LABEL(norenorm)
  // Here we have to do something fiddly. Renormalisation would be a trivial
  // job if we had the leading mantissa bit - just note that it's one bit
  // position above where it should be, and shift right by one. But without
  // that bit, we currently have (2x - 2^30), and we want (x - 2^30); just
  // shifting right would of course give us (x - 2^29), so we must subtract an
  // extra 2^29 to fix this up.
  lsrs    r3, r3, #1
  movs    r1, #1
  lsls    r1, r1, #29
  subs    r3, r3, r1
  adds    r2, r2, #1
LOCAL_LABEL(norenorm):
  // Round and shift down to the right bit position.
  lsrs    r0, r3, #7    // round bit goes into the carry flag
  bcc     LOCAL_LABEL(rounded)
  adds    r0, r0, #1
  // In the round-up branch, we must also check if we have to round to even, by
  // testing all the bits below the round bit. We will normally not expect to,
  // so we do RTE by branching out of line and back again to avoid spending a
  // branch in the common case.
  lsls    r5, r3, #32-7+1  // check the bits shifted out of r3 above
  bne     LOCAL_LABEL(rounded)          // if any is nonzero, we're not rounding to even
  lsls    r5, r6, #15      // check the bottom 17 bits of the low-order 32
                           //   (enough to overlap r3 even if we renormalised)
  beq     LOCAL_LABEL(rte)              // if any is nonzero, fall through, else RTE
LOCAL_LABEL(rounded):
  // Put on the sign and exponent, check for underflow and overflow, and
  // return.
  //
  // Underflow occurs iff r2 (the output exponent) <= 0. Overflow occurs if
  // it's >= 0xFF. (Also if it's 0xFE and we rounded up to overflow, but since
  // this code doesn't report exceptions, we can ignore this case because it'll
  // happen to return the right answer regardless). So we handle most of this
  // via an unsigned comparison against 0xFF, which leaves the one case of a
  // zero exponent that we have to filter separately by testing the Z flag
  // after we shift the exponent back up into place.
  cmp     r2, #0xFF    // check for most over/underflows
  bhs     LOCAL_LABEL(outflow)      // ... and branch out of line for them
  lsls    r5, r2, #23  // shift the exponent into its output location
  beq     LOCAL_LABEL(outflow)      // ... and branch again if it was 0
  lsls    r4, r4, #31  // shift the output sign into place
  orrs    r0, r0, r4   // and OR it in to the output
  adds    r0, r0, r5   // OR in the mantissa
  pop     {r4,r5,r6,pc} // and return

LOCAL_LABEL(rte):
  // Out-of-line handler for the round-to-even case. Clear the low mantissa bit
  // and go back to the post-rounding code.
  movs    r5, #1
  bics    r0, r0, r5
  b       LOCAL_LABEL(rounded)

LOCAL_LABEL(outflow):
  cmp     r2, #0
  bgt     LOCAL_LABEL(overflow)
  // To handle underflow, we construct an intermediate value in the IEEE 754
  // style (using our existing full-length mantissa, and bias the exponent by
  // +0xC0), and indicate whether that intermediate was rounded up, down or not
  // at all. Then call the helper function funder, which will denormalise and
  // re-round correctly.
  lsls    r1, r0, #7    // shift up the post-rounding mantissa
  subs    r1, r3, r1    //   and subtract it from the pre-rounding version
  lsls    r6, r6, #15
  cmp     r6, #1        // if the rest of the low bits are nonzero
  adcs    r1, r1, r1    //   then set an extra bit at the bottom

  lsls    r4, r4, #31
  orrs    r0, r0, r4    // put on the sign
  adds    r2, r2, #192  // bias the exponent
  lsls    r3, r2, #23
  adds    r0, r0, r3    // put on the biased exponent

  bl      SYMBOL_NAME(__compiler_rt_funder)
  pop     {r4,r5,r6,pc}

LOCAL_LABEL(overflow):
  // Handle overflow by returning an infinity of the correct sign.
  lsls    r4, r4, #8    // move the sign up to bit 8
  movs    r0, #0xff
  orrs    r0, r0, r4    // fill in an exponent just below it
  lsls    r0, r0, #23   // and shift those 9 bits up to the top of the word
  pop     {r4,r5,r6,pc}

  // We come here if there's at least one zero or denormal. On the fast path
  // above, it was convenient to check these before checking NaNs and
  // infinities, but NaNs take precedence, so now we're off the fast path, we
  // must still check for those.
  //
  // At the main entry point 'zerodenorm' we want r2 and r3 to be the two input
  // exponents. So if we branched after shifting-and-checking r2, we come to
  // this earlier entry point 'zerodenorm0' so that we still shift r3.
LOCAL_LABEL(zerodenorm0):
  lsrs    r3, r3, #24
LOCAL_LABEL(zerodenorm):
  cmp     r2, #255
  beq     LOCAL_LABEL(naninf)
  cmp     r3, #255
  beq     LOCAL_LABEL(naninf)
  // Now we know we have at least one zero or denormal, and no NaN or infinity.
  // Check if either input is actually zero. We've ruled out 0 * infinity by
  // this point, so any zero input means we return zero of the correct sign.
  lsls    r6, r0, #1        // is one input zero?
  beq     LOCAL_LABEL(zero)              // yes, go and return zero
  lsls    r6, r1, #1        // is the other one zero?
  bne     LOCAL_LABEL(denorm)            // if not, one must have been a denormal
LOCAL_LABEL(zero):
  lsls    r0, r4, #31    // shift up the output sign to make the return value
  pop     {r4,r5,r6,pc}

  // Handle denormals via the helper function fnorm2, which will break both
  // inputs up into mantissa and exponent, renormalising and generating a
  // negative exponent if necessary.
LOCAL_LABEL(denorm):
  push    {r0,r1,r2,r3}
  mov     r0, sp
  bl      SYMBOL_NAME(__compiler_rt_fnorm2)
  pop     {r0,r1,r2,r3}
  // Convert fnorm2's return values into the right form to rejoin the main
  // code path.
  lsls    r0, r0, #1
  lsls    r1, r1, #1
  adds    r2, r2, r3
  subs    r2, r2, #0x7f
  b       LOCAL_LABEL(normalised)

  // We come here if at least one input is a NaN or infinity. There may still
  // be zeroes (or denormals, though they make no difference at this stage).
LOCAL_LABEL(naninf):
  movs    r6, #0xff
  lsls    r6, r6, #24
  lsls    r5, r0, #1
  cmp     r5, r6
  bhi     LOCAL_LABEL(nan)              // first operand is a NaN
  lsls    r5, r1, #1
  cmp     r5, r6
  bhi     LOCAL_LABEL(nan)              // second operand is a NaN

  // We know we have at least one infinity, and no NaNs. We might also have a
  // zero, in which case we return the default quiet NaN.
  lsls    r6, r0, #1
  beq     LOCAL_LABEL(infzero)          // if r0 is a zero, r1 must be inf
  lsls    r6, r1, #1
  beq     LOCAL_LABEL(infzero)          // if r1 is a zero, r0 must be inf
  // Otherwise we have infinity * infinity, or infinity * finite. Just return
  // an appropriately signed infinity.
  b       LOCAL_LABEL(overflow)         // reuse the code there

  // We come here if at least one input is a NaN. Hand off to fnan2, which
  // propagates an appropriate NaN to the output, dealing with the special
  // cases of signalling/quiet NaNs.
LOCAL_LABEL(nan):
  bl      SYMBOL_NAME(__compiler_rt_fnan2)
  pop     {r4,r5,r6,pc}

  // Return a quiet NaN as the result of infinity * zero.
LOCAL_LABEL(infzero):
  ldr     r0, =0x7fc00000
  pop     {r4,r5,r6,pc}

END_COMPILERRT_FUNCTION(__mulsf3)

NO_EXEC_STACK_DIRECTIVE

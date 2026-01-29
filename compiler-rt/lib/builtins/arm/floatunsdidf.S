//===-- floatunsdidf.S - 64-bit unsigned int to double-precision FP conversion//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __floatunsdidf function (64-bit unsigned integer to
// double precision floating point conversion), with the IEEE-754 default
// rounding (to nearest, ties to even), for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"
#include "endian.h"

  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__floatunsdidf)
  push {r4, lr}
  bl __aeabi_ul2d
  VMOV_TO_DOUBLE(d0, r0, r1)
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__floatunsdidf, __aeabi_ul2d)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_ul2d)

  // There are essentially three cases we need to separate. The leading bit of
  // the integer is either in xh or xl; if it's in xh, it makes a difference
  // whether it's above bit 20, because that's the case where we must shift
  // right and potentially round.
  //
  // Start by assuming the high word is nonzero; if we're wrong, we'll find out
  // in a few instructions' time and be able to try again. So we find the
  // position of the leading bit in xh, and turn it into a left-shift count
  // that will move the leading bit up to where it belongs in the output
  // double.
  clz     r3, xh
  subs    r3, r3, #11

  // If that left-shift count is negative, we're going to have to shift the
  // mantissa right instead of left, and maybe round it. Branch out of line for
  // the code that handles that case.
  blo     LOCAL_LABEL(shiftdown)

  // Shift xh left to bring the top word of the mantissa to the right place. By
  // making this shift set the flags, we detect if xh was zero.
  //
  // We branch out of line if it _wasn't_ zero, on the theory that small input
  // integers are likely to occur more often than large ones, so the small case
  // should be the faster path. This is a bit of a compromise between large and
  // small integer performance: if we wanted to prioritise small inputs above
  // all else, we could have tested if xh=0 to begin with - but that would cost
  // an extra instruction on the large-integer path, because it repeats work
  // that this instruction can do in passing.
  lsls    xh, xh, r3
  bne     LOCAL_LABEL(highword)

  // Now we've found out that xh=0, we need to repeat the CLZ instruction on
  // xl. The simplest thing is to shift xl up by a variable distance to put its
  // leading bit at the top; then we can do immediate shifts to move it up
  // further to the top of the double-precision mantissa. (Otherwise you'd have
  // to make a second shift count by subtracting from 32, using more registers
  // and requiring more register-controlled shifts, especially awkward in
  // Thumb.)
  //
  // There may not _be_ a leading bit in xl at all (just as there turned out
  // not to have been one in xh, if we're on this path). In that case the input
  // integer was 0, and so we should return double-precision 0, which
  // conveniently has the same representation (xh=xl=0 already).
  clz     r3, xl                  // decide how far to shift up
  lsls    xh, xl, r3              // do the shift, also checking if xl = 0
  bxeq    lr                      // if xl = 0, return zero immediately

  // Now xl contains the output mantissa, with the leading bit at the top. We
  // must shift that up another 21 bits, and recombine it with an exponent
  // derived from r3 (telling us how far we've already shifted up).
  //
  // If r3=0 then the input value was in the range [2^31,2^32), so its exponent
  // in double precision should be 0x41e. We want to reduce that by 1 so that
  // the leading bit of the mantissa will increment it when we add it in. So
  // the exponent should be 0x41d minus r3.
  rsb     r3, r3, #0x1d           // 0x1d minus shift count
  add     r3, r3, #0x400          // 0x41d minus shift count
  lsr     r2, xh, #11             // make top word of mantissa
  lsl     xl, xh, #21             // make bottom word of mantissa
  add     xh, r2, r3, lsl #20     // and combine it with exponent
  bx      lr

LOCAL_LABEL(highword):
  // This is the branch for numbers big enough that xh != 0, but not big enough
  // to need to shift downwards and round.
  //
  // r3 is the distance that we've already shifted xh left by. We'll need to
  // shift xl left by the same amount, and we'll also need to shift xl right by
  // 32 minus that, to put some of its bits at the bottom of xh.
  rsb     r12, r3, #32
#if !__thumb__
  orr     xh, xh, xl, lsr r12
#else
  // In Thumb we have to do the register-controlled shift and the OR in
  // separate instructions.
  lsr     r12, xl, r12
  orr     xh, xh, r12
#endif
  // Shift xl left as well, so that xh:xl are now the full output mantissa,
  // with its leading bit in bit 20 of xh.
  lsls    xl, xl, r3

  // Calculate the exponent, and recombine it with the mantissa. This is
  // exactly the same method as above, except that the exponent is different,
  // because this time r3 stores the offset between the original leading bit
  // position and bit 20 of the mantissa, so that it's zero if the input is in
  // the range [2^52,2^53), which would make the output exponent 0x433, or
  // 0x432 after compensating for the leading mantissa bit.
  rsb     r3, r3, #0x32           // 0x32 minus shift count
  add     r3, r3, #0x400          // 0x432 minus shift count
  add     xh, xh, r3, lsl #20     // combine with the top word of the mantissa
  bx      lr

LOCAL_LABEL(shiftdown):
  // This is the branch for numbers so big that the mantissa has to be shifted
  // _right_, so that some of the mantissa is shifted off the bottom and the
  // number has to be rounded.
  //
  // r3 contains the shift count, but it's currently negative (it was
  // calculated as a left shift). So it's in a good state to use for
  // calculating the output exponent, and therefore we do that first, while
  // it's convenient.
  rsb     r2, r3, #0x32           // 0x32 minus shift count
  add     r2, r2, #0x400          // 0x432 minus shift count

  // Shift the mantissa down to the right position, capturing the bits shifted
  // off the bottom at the top of r3. We'll need to temporarily push a couple
  // of extra registers for this part, because we need to calculate how far to
  // shift xh and xl right, but also how far to shift them left to get the bits
  // shifted out of each one.
  push    {r4,lr}
  rsb     r4, r3, #0              // r4 = right-shift count
  rsb     lr, r4, #32             // lr = left-shift count
  lsl     r12, xh, lr             // r12 = bits shifted out of xh
  lsr     xh, xh, r4              // shift xh right to make its final value
  lsl     r3, xl, lr              // r3 = bits shifted out of xl
#if !__thumb__
  orrs    xl, r12, xl, lsr r4     // shift xl right and combine with r12
#else
  // In Thumb we have to do the register-controlled shift and the OR in
  // separate instructions.
  lsrs    xl, xl, r4
  orr     xl, xl, r12
#endif
  pop     {r4,lr}

  // Now xh:xl contains the unrounded output mantissa; r2 contains its
  // exponent; and r3 contains the bits shifted off the bottom. Also, the
  // single flag-setting shift in the sequence above was the one that shifted
  // xl right, so the carry flag contains the bit just off the bottom, i.e. the
  // bit that tells us whether we need to round up.
  //
  // Recombine the mantissa with the exponent, and then if C is clear, we don't
  // need to round up, and can return.
  add     xh, xh, r2, lsl #20     // put back the exponent
  bxcc    lr                      // return if we don't have to round

  // We're rounding up, and we may also need to round to even.
  adds    xl, xl, #1              // increment the mantissa to round up
  adc     xh, xh, #0              //   and propagate a carry if any
  lsls    r3, r3, #1              // set Z if we had an exact halfway case
  biceq   xl, xl, #1              //   and round back to even if so
  bx      lr

END_COMPILERRT_FUNCTION(__aeabi_ul2d)

NO_EXEC_STACK_DIRECTIVE

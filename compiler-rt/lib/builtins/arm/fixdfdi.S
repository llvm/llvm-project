//===-- fixdfdi.S - double-precision FP to 64-bit signed int conversion ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __fixdfdi function (double precision floating point
// to 64-bit signed integer conversion), with the C rounding semantics
// (rounding towards zero), for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"
#include "endian.h"

  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__fixdfdi)
  push {r4, lr}
  VMOV_FROM_DOUBLE(r0, r1, d0)
  bl __aeabi_d2lz
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__fixdfdi, __aeabi_d2lz)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_d2lz)

  // The fast path: deliver an answer as quickly as possible for positive cases
  // that don't overflow, and branch out of line to handle everything else more
  // slowly, including negative numbers, overflows, and NaNs.
  //
  // The basic idea is to make a bare version of the mantissa, with its leading
  // 1 bit explicit at the top of the word, and shift it right by an amount
  // derived from the exponent.

  // Shift the exponent down to the bottom of the word. Using ASR here in place
  // of LSR means the sign bit keeps its original value, and therefore as a
  // side effect the N flag will tell us whether the input was negative. On the
  // positive path the output is the same anyway.
  asrs    r2, xh, #20

  // Make the 64-bit mantissa word, with its top half in r3 and the bottom half
  // in r12, and set the leading mantissa bit at the top of r3.
  //
  // (In principle, the leading mantissa bit shouldn't be set at all if the
  // input exponent is zero, because then the input is either 0 or a denormal.
  // But since we're always rounding towards zero and not delivering any output
  // indicating whether the result is inexact, it makes no difference - all
  // denormals will round down to 0 anyway, and will still do so even if we
  // imagine that their leading mantissa bit was set.)
  lsl     r3, xh, #11           // shift top word left to discard sign+exponent
  orr     r3, r3, xl, lsr #21   // combine with bits from bottom word
  orr     r3, r3, #1<<31        // set the leading bit
  lsl     r12, xl, #11          // shift bottom word left

  // Branch out of line for negative cases, using the N flag set by the ASRS at
  // the start of the function.
  //
  // For 64-bit outputs it's difficult to handle both signs in a shared code
  // path. Converting to a 32-bit integer you can finish up with an RSBMI to
  // conditionally negate it, but negating a 64-bit number requires two
  // instructions and clobbers the flags in between, so you can't
  // conditionalize both instructions on the same prior state of the flags.
  bmi     LOCAL_LABEL(neg)

  // Convert the mantissa into a right shift count, indicating how many bits we
  // want to shift the 64-bit mantissa in r3:r12 right by.
  //
  // If that's negative or zero, that means the input is too big, or the
  // exponent is 0x7FF (so we might have a NaN), so branch out of line again.
  //
  // Another possibility is that if the number is very small, the right shift
  // count might not fit in a byte, in which case the AArch32 shift semantics
  // might do the wrong thing, since they only look at the low byte.
  //
  // To spot this efficiently, we do the conversion in two steps, arranged so
  // that the first step detects massive underflow and the second detects
  // overflow. We needed two steps anyway, because we need to subtract the
  // input exponent from 0x43e (the exponent of input numbers between 2^63 and
  // 2^64), and that value doesn't fit in an immediate field. So the first step
  // subtracts 0x340, which makes r2 become negative if the exponent is very
  // small; then the second step subtracts from 0xfe, giving 0xfe-(exp-0x340))
  // = 0x43e-exp.
  //
  // The input is tiny if r2 is negative after the first subtraction. In that
  // situation we set r2 to 0 before the second step, treating all exponents
  // smaller than 0x340 as if they had been 0x340. We do this by ASR+BIC, which
  // avoids needing an IT instruction in Thumb.
  sub     r2, r2, #0x340        // first stage: r2 is negative for underflow
  bic     r2, r2, r2, asr #31   // if so, saturate by clearing all bits of r2
  rsbs    r2, r2, #0xfe         // second step: r2 is now the right shift count
  ble     LOCAL_LABEL(invalid)          // if it was negative or zero, overflow

  // Make the top word of the result, which is the easy part: it's a
  // shifted-right version of the top word of the mantissa, which will be zero
  // if the shift count is 32 or more.
  lsrs    xh, r3, r2

  // Make the bottom word of the result. If the shift count is less than 32,
  // this will be (ml >> n) | (mh << (32-n)), where 'mh' and 'ml' denote the
  // high and low words of the mantissa (in r3 and r12 respectively). Otherwise
  // the whole high output word is zero, and the low mantissa word contributes
  // nothing to the output, so we just need mh >> (n-32).
  lsr     xl, r12, r2           // start by setting it to ml >> n
  rsbs    r2, r2, #32           // compute 32-n, and find which case we're in
#if !__thumb__
  orrhi   xl, xl, r3, lsl r2    // shift count < 32, so OR in ml << (32-n)
#else
  // In Thumb we must separate the register-controlled shift and the OR into
  // two instructions.
  lslhi   r3, r3, r2
  orrhi   xl, xl, r3
#endif
  rsbls   r2, r2, #0            // shift count > 32, so compute n-32
  lsrls   xl, r3, r2            // replace xl with mh >> (n-32)

  // We're done: the result is already rounded towards zero.
  bx      lr

LOCAL_LABEL(neg):
  // Handle negative numbers. We come here with the mantissa already prepared
  // in r3 and r12, and the exponent in the bottom 11 bits of r2 with all 1s
  // above it (because it was shifted down via ASR and the sign bit was set).
  // In other words, r2 = exponent - 0x800.

  // Convert the exponent into a shift count, exactly as on the positive path
  // except that the initial subtraction of 0x340 is adjusted to cancel out the
  // offset of 0x800 in r2, so that it becomes an addition.
  add     r2, r2, #0x800-0x340  // first stage: r2 is negative for underflow
  bic     r2, r2, r2, asr #31   // if so, saturate by clearing all bits of r2
  rsbs    r2, r2, #0xfe         // second step: r2 is now the right shift count
  ble     LOCAL_LABEL(invalid)          // if it was negative or zero, overflow

  // Shift the mantissa down to create the absolute value of the result in
  // xh:xl, exactly as on the positive path above.
  lsrs    xh, r3, r2            // make high word of mantissa
  lsr     xl, r12, r2           // initially set low word to ml >> n
  rsbs    r2, r2, #32           // compute 32-n, and find which case we're in
#if !__thumb__
  orrhi   xl, xl, r3, lsl r2    // shift count < 32, so OR in ml << (32-n)
#else
  // In Thumb we must separate the register-controlled shift and the OR into
  // two instructions.
  lslhi   r3, r3, r2
  orrhi   xl, xl, r3
#endif
  rsbls   r2, r2, #0            // shift count > 32, so compute n-32
  lsrls   xl, r3, r2            // and replace xl with mh >> that

  // All of that was almost exactly the same as the positive code path, but now
  // we must negate the answer before returning it.
  rsbs    xl, xl, #0            // negate low word, setting carry flag
#if !__thumb__
  rsc     xh, xh, #0            // negate high word
#else
  // Thumb has no RSC, so substitute MVN + ADC.
  mvn     xh, xh
  adc     xh, xh, #0
#endif

  // We're done.
  bx      lr

LOCAL_LABEL(invalid):
  // We come here if the exponent field of the number is large enough that it's
  // either a NaN or infinity, or a finite number of absolute value at least
  // 2^63.
  //
  // For out-of-range positive values, we return the maximum positive signed
  // integer 0x7fffffffffffffff. For out-of-range negative values, we return
  // the minimum negative signed integer 0x8000000000000000. For NaNs, we
  // return zero.
  //
  // Not _every_ number of this kind is actually an invalid input. The exact
  // value -2^63 is perfectly valid. If this implementation supported FP
  // exceptions, we'd have to detect that one case and return
  // 0x8000000000000000 with no exception, while raising an Invalid Operation
  // exception for everything else. But since we don't support exceptions, we
  // don't have to tell the difference here: -2^63 and negative overflows both
  // return 0x8000000000000000, and it doesn't matter that one is the right
  // answer and the other a best-effort error response.

  // Check for NaNs and branch out of line.
  cmp     xl, #1                  // set C if any bit of xl is nonzero
  adc     r2, xh, xh              // shift that bit into xh, discarding sign
  cmn     r2, #1 << 21            // then greater than 0xFFE00000 means NaN
  bhi     LOCAL_LABEL(nan)

  // The remaining case is a too-large exponent. Return either INT_MAX or
  // INT_MIN depending on sign, using the fact that (xh ASR 31) is 0 for a
  // positive input or 0xFFFFFFFF for a negative input, so it's exactly the
  // bitwise inverse of the correct low word of the output.
  mvn     xl, xh, asr #31         // make the low word of the output
  eor     xh, xl, #0x80000000     // make the high word by flipping sign bit
  bx      lr

LOCAL_LABEL(nan):
  // Return zero, for use when the input was a NaN.
  movs    xh, #0
  movs    xl, #0
  bx      lr

END_COMPILERRT_FUNCTION(__aeabi_d2lz)

NO_EXEC_STACK_DIRECTIVE

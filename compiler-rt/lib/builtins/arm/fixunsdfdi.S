//===-- fixunsdfdi.S - double-precision FP to 64-bit unsigned int conversion==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __fixunsdfdi function (double precision floating
// point to 64-bit unsigned integer conversion), with the C rounding semantics
// (rounding towards zero), for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"
#include "endian.h"

  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__fixunsdfdi)
  push {r4, lr}
  VMOV_FROM_DOUBLE(r0, r1, d0)
  bl __aeabi_d2ulz
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__fixunsdfdi, __aeabi_d2ulz)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_d2ulz)

  // The fast path: deliver an answer as quickly as possible for positive cases
  // that don't overflow, and branch out of line to handle everything else more
  // slowly, including negative numbers, overflows, and NaNs.
  //
  // The basic idea is to make a bare version of the mantissa, with its leading
  // 1 bit explicit at the top of the word, and shift it right by an amount
  // derived from the exponent.

  // Shift the exponent down to the bottom of the word. We keep the sign bit,
  // which will cause negative inputs to be treated like positive numbers with
  // extra-enormous exponent. This is good enough to trigger a branch out of
  // line later without needing an extra instruction, and then we can handle
  // negative inputs properly after that branch.
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

  // Convert the mantissa into a right shift count, indicating how many bits we
  // want to shift the 64-bit mantissa in r3:r12 right by.
  //
  // If that's negative, that means the input is too big, or the exponent is
  // 0x7FF (so we might have a NaN), or the value in r2 was 0x800 or more
  // because the input was negative, so branch out of line again.
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
  blt     LOCAL_LABEL(uncommon)        // if it was negative, overflow

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

LOCAL_LABEL(uncommon):
  // We come here if the exponent field of the number is either too large or
  // too small, or if the sign bit is set, so that the number is one of
  //  - a NaN
  //  - an infinity
  //  - a positive finite number of absolute value at least 2^31
  //  - any negative number at all.

  // Test the sign bit first, and branch out of line for positive values, on
  // the theory that reasonably sensible negative values (like -0.5) are more
  // likely than the overflow, infinity or NaN cases, and want to be handled as
  // quickly as possible.
  tst     xh, xh
  bpl     LOCAL_LABEL(positive_invalid)

  // Now we have either a negative finite value, -inf, or a NaN (with the sign
  // bit set). Conveniently, _all_ of those just return 0, so we don't have to
  // bother checking which.
  //
  // If we were reporting exceptions, then inputs in the range (-1,0] would
  // _legally_ return 0 (after rounding toward zero), whereas -1 and below
  // would raise the IEEE Invalid Operation exception.
LOCAL_LABEL(return_zero):
  movs    xh, #0
  movs    xl, #0
  bx      lr

LOCAL_LABEL(positive_invalid):
  // Here we have a positive value causing an invalid operation exception. If
  // it's a NaN, we return zero; otherwise we return UINT_MAX.
  cmp     xl, #1                  // set C if any bit of xl is nonzero
  adc     r2, xh, xh              // shift that bit into xh, discarding sign
  cmn     r2, #1 << 21            // then greater than 0xFFE00000 means NaN
  bhi     LOCAL_LABEL(return_zero)       // so we go back to the case above

  mov     xh, #0xFFFFFFFF
  mov     xl, #0xFFFFFFFF
  bx      lr

END_COMPILERRT_FUNCTION(__aeabi_d2ulz)

NO_EXEC_STACK_DIRECTIVE

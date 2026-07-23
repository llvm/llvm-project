//===-- divsf3.S - single-precision floating point division ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements single-precision soft-float division with the IEEE-754
// default rounding (to nearest, ties to even), in optimized AArch32 assembly
// language suitable to be built as either Arm or Thumb2.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"


  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__divsf3)
  push {r4, lr}
  vmov r0, s0
  vmov r1, s1
  bl __aeabi_fdiv
  vmov s0, r0
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__divsf3, __aeabi_fdiv)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_fdiv)
  // Extract the exponents of the inputs into r2 and r3, occupying bits 16-23
  // of each register so that there will be space lower down to store extra
  // data without exponent arithmetic carrying into it. In the process, check
  // both exponents for 00 or FF and branch out of line to handle all the
  // uncommon types of value (infinity, NaN, zero, denormals).
  //
  // Chaining conditional instructions like this means that the second
  // instruction (setting up r3) might not be executed at all, so fdiv_uncommon
  // will have to redo it just in case. That saves an instruction here,
  // executed for _all_ inputs, and moves it to the uncommon path run for only
  // some inputs.
  mov     r12, #0xFF0000
  ands    r2, r12, r0, lsr #7   // r2 has exponent of numerator. (Is it 0?)
  andsne  r3, r12, r1, lsr #7   // r3 has exponent of denominator. (Is it 0?)
  teqne   r2, r12               // if neither was 0, is one FF?
  teqne   r3, r12               // or the other?
  beq     LOCAL_LABEL(uncommon)         // branch out of line if any answer was yes

  // Calculate the output sign, which is always just the XOR of the input
  // signs. Store it in bit 8 of r2, below the numerator exponent.
  teq     r0, r1                // is the output sign bit 1?
  orrmi   r2, r2, #0x100        // if so, set bit 8 of r2

  // Isolate the mantissas of both values, by setting bit 23 of each one and
  // clearing the 8 bits above that.
  //
  // In the process, swap the register allocations (which doesn't cost extra
  // instructions if we do it as part of this manipulation). We want the
  // numerator not to be in r0, because r0 is where we'll build up the quotient
  // while subtracting things from the numerator.
  orr     r12, r0, #1 << 23
  orr     r0, r1, #1 << 23
  bic     r1, r12, #0xFF000000
  bic     r0, r0, #0xFF000000

LOCAL_LABEL(div):
  // Start of the main division. We get here knowing that:
  //
  //   r0 = mantissa of denominator, with the leading 1 at bit 23
  //   r1 = mantissa of numerator, similarly
  //   r2 = (exponent of numerator << 16) + (result sign << 8)
  //   r3 = (exponent of denominator << 16)

  push    {r14}                 // we'll need an extra register

  // Calculate the initial result exponent by just subtracting the two input
  // exponents. This doesn't affect the sign bit lower down in r2.
  sub     r2, r2, r3

  // That initial exponent might need to be adjusted by 1, depending on whether
  // dividing the mantissas gives a value >=1 or <1. We don't need to wait
  // until the division is finished to work that out: we can tell immediately
  // by just comparing the mantissas.
  //
  // The basic idea is to do the comparison in a way that sets the C flag if
  // numerator >= denominator. Then we recombine the sign and exponent by doing
  // "ADC r2, r2, r2, asr #16": the exponent in the top half of r2 is shifted
  // down to the low 8 bits, just below the sign bit, and using ADC rather than
  // ADD folds in the conditional increment from the mantissa comparison.
  //
  // If we're not incrementing the output exponent, we instead shift the
  // numerator mantissa left by 1, so that it _is_ greater than the denominator
  // mantissa. Otherwise we'd generate only a 22-bit quotient, instead of 23.
  //
  // The exponent also needs to be rebiased, so that dividing two numbers the
  // same gives an output exponent of 0x7F. If the two inputs have the same
  // exponent then we'll have computed an exponent of 0 via the SUB instruction
  // above; if the mantissas are the same as well then the ADC will increment
  // it; also, the leading bit of the quotient will increment the exponent
  // again when we recombine it with the output mantissa later. So we need to
  // add (0x7F - 2) to the mantissa now, to make an exponent of 0 from the SUB
  // come to 0x7F after both of those increments.
  //
  // Putting all of that together, what we _want_ to do is this:
  //
  // [#1]   CMP     r1, r0                // set C if num >= den
  // [#2]   MOVLO   r1, r1, lsl #1        // if num < den, shift num left
  // [#3]   ADD     r2, r2, #0x7D0000     // rebias exponent
  // [#4]   ADC     r2, r2, r2, asr #16   // combine sign + exp + adjustment
  //
  // However, we only do the first of those four instructions right here. The
  // other three are distributed through the code below, after unrelated load
  // or multiply instructions which will have a result delay slot on simple
  // CPUs. Each is labelled "exponent setup [#n]" in a comment.
  //
  // (Since instruction #4 depends on the flags set up by #2, we must avoid
  // clobbering the flags in _any_ of the instructions interleaved with this!)
  cmp     r1, r0                // exponent setup [#1]

  // Start the mantissa division by making an approximation to the reciprocal
  // of the denominator. We first obtain an 8-bit approximation using a table
  // lookup indexed by the top 7 denominator bits (counting the leading 1, so
  // really there are only 6 bits in the table index).
  //
  // (r0 >> 17) is the table index, and its top bit is always set, so it ranges
  // from 64 to 127 inclusive. So we point the base register 64 bytes before
  // the actual table.
  adr     r12, LOCAL_LABEL(tab) - 64
#if __thumb__
  // Thumb can't do this particular shift+add+load in one instruction - it only
  // supports left shifts of 0 to 3 bits, not right shifts of 17. So we must
  // calculate the load offset separately.
  add     r14, r12, r0, lsr #17
  ldrb    r14, [r14]
#else
  ldrb    r14, [r12, r0, lsr #17]
#endif

  // Now do an iteration of Newton-Raphson to improve that 8-bit approximation
  // to have 15-16 accurate bits.
  //
  // Basics of Newton-Raphson for finding a reciprocal: if you want to find 1/d
  // and you have some approximation x, your next approximation is X = x(2-dx).
  // Looked at one way, this is the result of applying the N-R formula
  // X=x-f(x)/f'(x) to the function f(x) = 1/x - d. Another way to look at it
  // is to suppose that dx = 1 - e, for some e which is small (because dx is
  // already reasonably close to 1). Then you want to double the number of
  // correct bits in the next approximation, i.e. square the error. So you want
  // dX = 1-e^2 = (1-e)(1+e) = dx(2-dx). Cancelling d gives X = x(2-dx) again.
  //
  // In this situation, we're working in fixed-point integers rather than real
  // numbers, and all the scales are different:
  //  * our input denominator d is in the range [2^23,2^24)
  //  * our input approximation x is in the range [2^7,2^8)
  //  * we want the output approximation to be in the range [2^15,2^16)
  // Those factors combine to mean that we want
  //   x(2^32-dx) / 2^23
  // = (2^9 x) - (dx^2 / 2^23)
  //
  // But we also want to compute this using ordinary MUL, not a long multiply
  // instruction (those are slower). So we need to worry about the product
  // overflowing. dx fits in 32 bits, because it's the product of something
  // <2^24 with something <2^8; but we must shift it right before multiplying
  // by x again.

  mul     r12, r0, r14          // r12  = dx
  movlo   r1, r1, lsl #1        //   exponent setup [#2] in the MUL delay slot
  mvn     r12, r12, lsr #8      // r12 ~= -dx/2^8
  mul     r3, r12, r14          // r3  ~= -dx^2/2^8
  mov     r14, r14, lsl #9      // r14  = 2^9 x
  add     r14, r14, r3, asr #15 // r14 ~= 2^9 x - dx^2 / 2^23

  // Now r14 is a 16-bit approximation to the reciprocal of the input mantissa,
  // scaled by 2^39 (so that the min mantissa 2^23 would have reciprocal 2^16
  // in principle, and the max mantissa 2^24-1 would have reciprocal just over
  // 2^15). The error is always negative (r14 is an underestimate of the true
  // value), and the maximum error is 6 and a bit ULP (that is, the true
  // reciprocal is strictly less than (r14+7)). Also, r14 is always strictly
  // less than 0x10000 (even in the case of the min mantissa, where the true
  // value would be _exactly_ 0x10000), which eliminates a case of integer
  // overflow.
  //
  // All of these properties of the reciprocal approximation are checked by
  // exhaustively iterating over all 2^23 possible input mantissas. (The nice
  // thing about doing this in single rather than double precision!)
  //
  // Now we extract most of the quotient by two steps of long division, using
  // the reciprocal estimate to identify a multiple of the denominator to
  // subtract from the numerator. To avoid integer overflow, the numerator
  // mantissa is shifted down 8 bits so that it's less than 0x10000. After we
  // calculate an approximate quotient, we shift the numerator left and
  // subtract that multiple of the denominator, moving the next portion of the
  // numerator into range for the next iteration.

  // First iteration of long division. We shift the numerator left 11 bits, and
  // since the quotient approximation is scaled by 2^31, we must shift that
  // right by 20 to make the right product to subtract from the numerator.
  mov     r12, r1, lsr #8       // shift the numerator down
  mul     r12, r14, r12         // make the quotient approximation
  mov     r1, r1, lsl #11       // shift numerator left, ready for subtraction
  mov     r3, r12, lsr #20      // make first 12-bit block of quotient bits
  mls     r1, r0, r3, r1        // subtract that multiple of den from num

  add     r2, r2, #0x7D0000     //   exponent setup [#3] in the MLS delay slot

  // Second iteration of long division. Differences from the first step: this
  // time we shift the numerator 12 bits instead of 11, so that the total of
  // both steps is 23 bits, i.e. we've shifted up by exactly the full width of
  // the output mantissa. Also, the block of output quotient bits is left in a
  // different register: it was in r3 the first time, and this time it's in
  // r12, so that we still have both available at the end of the process.
  mov     r12, r1, lsr #8       // shift the numerator down
  mul     r12, r14, r12         // make the quotient approximation
  mov     r1, r1, lsl #12       // shift numerator left, ready for subtraction
  mov     r12, r12, lsr #19     // make second 11-bit block of quotient
  mls     r1, r0, r12, r1       // subtract that multiple of den from num

  adc     r2, r2, r2, asr #16   //   exponent setup [#4] in the MLS delay slot

  // Now r1 contains the original numerator, shifted left 23, minus _some_
  // multiple of the original denominator (which is still in r0). The bounds on
  // the error in the above steps should make the error at most 1: that is, we
  // may have to subtract the denominator one more time to make r1 < r0, and
  // increment the quotient by one more.
  //
  // Our quotient is still in two pieces, computed separately in the above long
  // division steps. We fold the final increment into the same instruction that
  // recombines them, by doing the comparison in such a way that it sets the
  // carry flag if the increment is needed.

  cmp     r1, r0                // Set carry flag if num >= den
  subhs   r1, r1, r0            // If so, subtract den from num
  adc     r3, r12, r3, lsl #12  // Recombine quotient halves, plus optional +1

  // We've finished with r14 as a temporary register, so we can unstack it now.
  pop     {r14}

  // Now r3 contains the _rounded-down_ output quotient, and r1 contains the
  // remainder. That is, (denominator * r3 + r1) = (numerator << 23), and
  // 0 <= r1 < denominator.
  //
  // Next we must round to nearest, by checking if r1 is greater than half the
  // denominator. In division, it's not possible to hit an exact round-to-even
  // halfway case, so we don't need to spend any time checking for it.
  //
  // Proof of no round-to-even: define the 'width' of a dyadic rational to be
  // the distance between the lowest and highest 1 bits in its binary
  // representation, or equivalently, the index of its high bit if you scale it
  // by a power of 2 to make it an odd integer. E.g. any actual power of 2 has
  // width 0, and all of 0b11110, 0b1111, 0b11.11 and 0b0.01111 have width 3.
  // Then for any dyadic rationals a,b, width(ab) >= width(a)+width(b). Let w
  // be the maximum width that the input precision supports (so that for single
  // precision, w=23). Then if some division n/d were a round-to-even case, the
  // true quotient q=n/d would have width exactly w+1. But we have qd=n, so
  // width(n) >= width(q)+width(d) > w, which can't happen, because n is in the
  // input precision, hence had width <= w.)
  //
  // So we don't need to check for an exact _halfway_ case and clear the low
  // bit of the quotient after rounding up, as addition and multiplication both
  // need to do. But we do need to remember if the quotient itself was exact,
  // that is, if there was no remainder at all. That's needed in underflow
  // handling.

  // The rounding check wants to compare remainder with denominator/2. But of
  // course in integers it's easier to compare 2*remainder with denominator. So
  // we start by shifting the remainder left by 1, and in the process, set Z if
  // it's exactly 0 (i.e. the result needs no rounding at all).
  lsls    r1, r1, #1
  // Now trial-subtract the denominator. We don't do this at all if the result
  // was exact. If we do do it, r1 goes negative precisely if we need to round
  // up, which sets the C flag. (The previous instruction will have left C
  // clear, since r1 had its top 8 bits all clear. So now C is set _only_ if
  // we're rounding up.)
  subsne  r1, r1, r0
  // Recombine the quotient with the sign + exponent, and use the C flag from
  // the previous instruction to increment the quotient if we're rounding up.
  adc     r0, r3, r2, lsl #23

  // If we haven't either overflowed or underflowed, we're done. We can
  // identify most of the safe cases by doing an unsigned comparison of the
  // initial output exponent (in the top half of r2) with 0xFC: if 0 <= r2 <
  // 0xFC0000 then we have neither underflow nor overflow.
  //
  // Rationale: the value in the top half of r2 had three chances to be
  // incremented before becoming the exponent field of the actual output float.
  // It was incremented if we found the numerator mantissa was >= the
  // denominator (producing the value in the _bottom_ half of r2, which we just
  // ADCed into the output). Then it gets unconditionally incremented again
  // when the ADC combines it with the leading mantissa bit. And finally,
  // round-up might increment it a third time. So 0xFC is the smallest value
  // that can possibly turn into the overflowed value 0xFF after all those
  // increments.
  //
  // On the underflow side, (top half of r2) = 0 corresponds to a value of 1 in
  // the final result's exponent field (and then rounding might increase it
  // further); if the exponent was less than that then r2 wraps round and looks
  // like a very large positive integer from the point of view of this unsigned
  // comparison.
  cmp     r2, #0xFC0000
  bxlo    lr

  // The same comparison will have set the N and V flags to reflect the result
  // of comparing r2 with 0xFC0000 as a _signed_ integer. That reliably
  // distinguishes potential underflow (r2 is negative) from potential overflow
  // (r2 is positive and at least 0xFC0000)
  bge     LOCAL_LABEL(overflow)

  // Here we might or might not have underflow (but we know we don't have
  // overflow). To check more carefully, we look at the _bottom_ half of r2,
  // which contains the exponent after the first adjustment (for num >= denom),
  // That is, it's still off by 1 (compensating for the leading quotient bit),
  // and is also before rounding.
  //
  // We neglect the effect of rounding: division results that are tiny (less
  // than the smallest normalised number) before rounding, but then round up to
  // the smallest normal number, are an acceptable edge case to handle slowly.
  // We pass those to funder without worrying about them.
  //
  // So we want to check whether the bottom half of r2 was negative. It would
  // be nice to check bits 8-15 of it, but unfortunately, it's already been
  // combined with the sign (at bit 8), so those bits don't tell us anything
  // useful. Instead we look at the top 4 bits of the exponent field, i.e. the
  // 0xF0 bits. The largest _non_-overflowing exponent that might reach here is
  // less than 3, so it doesn't reach those bits; the smallest possible
  // underflow, obtained by dividing the smallest denormal by the largest
  // finite number, is -151 (before the leading bit increments it), which will
  // set the low 8 bits of r2 to 0x69. That is, the 0xF0 nibble of r2 will be
  // 0x60 or greater for a (pre-rounding) underflow, and zero for a
  // non-underflow.

  tst     r2, #0xF0
  bxeq    lr                    // no underflow after all; return

  // Rebias the exponent for funder, which also corrects the sign bit.
  add     r0, r0, #192 << 23
  // Tell funder whether the true value is greater or less than the number in
  // r0. This is obtained from the sign of the remainder (still in r1), with
  // the only problem being that it's currently reversed. So negate r1 (leaving
  // 0 at 0 to indicate exactness).
  rsbs    r1, r1, #0
  b     SYMBOL_NAME(__compiler_rt_funder)

LOCAL_LABEL(overflow):
  // Here we might or might not have overflow (but we know we don't have
  // underflow). We must check whether we really have overflowed.
  //
  // For this it's easiest to check the exponent field in the actual output
  // value in r0, after _all_ the adjustments have been completed. The largest
  // overflowed exponent is 0x193, and the smallest exponent that can reach
  // this is 0xFD (we checked against 0xFC above, but then the leading quotient
  // bit incremented it). So it's enough to shift the output left by one
  // (moving the exponent field to the top), increment it once more (so that
  // the smallest overflowed exponent 0xFF wraps round to 0), and then compare
  // against 0xFE000000 as an unsigned integer.
  mov     r12, r0, lsl #1
  add     r12, r12, #1 << 24
  cmp     r12, #0xFE << 24      // Check for exp = 253 or 254
  bxhs    lr
  // We have actual overflow. Rebias r0 to bring the exponent back into range,
  // which ensures its sign is correct. Then make an infinity of that sign to
  // return.
  subs    r0, r0, #0xC0 << 23
  movs    r12, #0xFF            // exponent of infinity
  orrs    r12, r12, r0, lsr #23 // exponent and sign at bottom of r12
  movs    r0, r12, lsl #23      // shift it up to the top of r0 to return
  bx      lr

LOCAL_LABEL(uncommon):
  // We come here from the start of the function if either input is an uncommon
  // value: zero, denormal, infinity or NaN.
  //
  // We arrive here with r12 = 0xFF000000, and r2 containing the exponent of x
  // in bits 16..23. But r3 doesn't necessarily contain the exponent of y,
  // because the instruction that set it up was conditional. So first we
  // unconditionally repeat it.
  and     r3, r12, r1, lsr #7

  // In all cases not involving a NaN as output, the sign of the output is made
  // in the same way as for finite numbers, as the XOR of the input signs. So
  // repeat the sign setup from the main branch.
  teq     r0, r1                // is the output sign bit 1?
  orrmi   r2, r2, #0x100        // if so, set bit 8 of r2

  // Detect infinities and NaNs, by checking if either of r2 or r3 is at least
  // 0xFF0000.
  cmp     r2, #0xFF0000
  cmplo   r3, #0xFF0000
  bhs     LOCAL_LABEL(inf_NaN)

  // Now we know there are no infinities or NaNs, but there's at least one zero
  // or denormal.
  movs    r12, r1, lsl #1       // is y zero?
  beq     LOCAL_LABEL(divbyzero)        // if so, go and handle division by zero
  movs    r12, r0, lsl #1       // is x zero? (now we know that y is not)
  moveq   r0, r2, lsl #23       // if so, 0/nonzero is just 0 (of right sign)
  bxeq    lr

  // Now we've eliminated zeroes as well, leaving only denormals: either x or
  // y, or both, is a denormal. Call fnorm2 to convert both into a normalised
  // mantissa and a (potentially small) exponent.
  and     r12, r2, #0x100       // save the result sign from r2
  lsr     r2, #16               // shift extracted exponents down to bit 0
  lsr     r3, #16               // where fnorm2 will expect them
  push    {r0, r1, r2, r3, r12, lr}
  mov     r0, sp                // tell fnorm2 where to find its data
  bl      SYMBOL_NAME(__compiler_rt_fnorm2)
  pop     {r0, r1, r2, r3, r12, lr}
  lsl     r3, #16               // shift exponents back up to bit 16
  orr     r2, r12, r2, lsl #16  // and put the result sign back in r2

  // Now rejoin the main code path, having finished the setup it will expect:
  // swap x and y, and shift the fractions back down to the low 24 bits.
  mov     r12, r0, lsr #8
  mov     r0, r1, lsr #8
  mov     r1, r12
  b       LOCAL_LABEL(div)

LOCAL_LABEL(inf_NaN):
  // We come here if at least one input is a NaN or infinity. If either or both
  // inputs are NaN then we hand off to fnan2 to propagate a NaN from the
  // input.
  mov     r12, #0xFF000000
  cmp     r12, r0, lsl #1       // if (r0 << 1) > 0xFF000000, r0 is a NaN
  blo     SYMBOL_NAME(__compiler_rt_fnan2)
  cmp     r12, r1, lsl #1
  blo     SYMBOL_NAME(__compiler_rt_fnan2)

  // No NaNs, so we have three options: inf/inf = NaN, inf/finite = inf, and
  // finite/inf = 0.

  // If both operands are infinity, we return a NaN. Since we know at
  // least _one_ is infinity, we can test this by checking if they're
  // equal apart from the sign bits.
  eor     r3, r0, r1
  lsls    r3, #1                // were all bits of XOR zero other than top?
  beq     LOCAL_LABEL(invalid)          // if so, both operands are infinity

  // See if x is infinite
  cmp     r12, r0, lsl #1       // (r0 << 1) == 0xFF000000?
  beq     LOCAL_LABEL(infret)           // if so, infinity/finite = infinity

  // y is infinite and x is not, so we return a zero of the
  // combined sign.
  eor     r0, r0, r1            // calculate the right sign
  and     r0, r0, #0x80000000   // throw away everything else
  bx      lr

LOCAL_LABEL(divbyzero):
  // Here, we know y is zero. But we don't know if x is zero or nonzero. So we
  // might be calculating 0/0 (invalid operation, generating a NaN), or
  // nonzero/0 (the IEEE "division by zero" exception, generating infinity).
  movs    r12, r0, lsl #1       // is x zero too?
  beq     LOCAL_LABEL(invalid)          // if so, go and return a NaN

LOCAL_LABEL(infret):
  // Here, we're either dividing infinity by a finite number, or dividing a
  // nonzero number by 0. (Or both, if we're dividing infinity by 0.) In all
  // these cases we return infinity with the sign from r2.
  //
  // If we were implementing IEEE exceptions, we'd have to separate these
  // cases: infinity / finite is not an _exception_, it just returns infinity,
  // whereas (finite and nonzero) / 0 is a division-by-zero exception. But here
  // we're not implementing exceptions, so we can treat all three cases the
  // same.
  //
  // r2 contains the output sign in bit 8, which is a convenient place to find
  // it when making an infinity, because we can fill in the 8 exponent bits
  // below that and then shift it left.
  orr     r2, r2, #0xff         // sign + maximum exponent
  lsl     r0, r2, #23           // shift up to the top
  bx      lr

LOCAL_LABEL(invalid):
  // Return the default NaN, from an invalid operation (either dividing
  // infinity by infinity, or 0 by 0).
  ldr     r0, =0x7FC00000
  bx      lr

// Finally, the lookup table for the initial reciprocal approximation.
//
// The table index is made from the top 7 bits of the denominator mantissa. But
// the topmost bit is always 1, so only the other 6 bits vary. So it only has
// 64 entries, not 128.
//
// Each table entry is a single byte, with its top bit set. So the table
// entries correspond to the reciprocal of a 7-bit mantissa prefix scaled up by
// 2^14, or the reciprocal of a whole 24-bit mantissa scaled up by 2^31.
//
// Each of these 64 entries corresponds to a large interval of possible
// mantissas. For example, if the top 7 bits are 1000001 then the overall
// mantissa could be anything from 0x820000 to 0x83FFFF. And because the output
// of this table provides more bits than the input, there are several choices
// of 8-bit reciprocal approximation for a number in that interval. The
// reciprocal of 0x820000 starts with 0xFC plus a fraction, and the reciprocal
// of 0x83FFFF starts with 0xF9 minus a fraction, so there are four reasonable
// choices for that table entry: F9, FA, FB or FC. Which do we pick?
//
// The table below is generated by choosing whichever value minimises the
// maximum possible error _after_ the approximation is improved by the
// Newton-Raphson step. In the example above, we end up with FA.
//
// The Python code below will regenerate the table, complete with the per-entry
// comments.

/*

for prefix in range(64, 128):
    best = None

    # Max and min 23-bit mantissas with this 7-bit prefix
    mmin, mmax = prefix * 2**17, (prefix + 1) * 2**17 - 1

    # Max and min table entry corresponding to the reciprocal of something in
    # that range of mantissas: round up the reciprocal of mmax, and round down
    # the reciprocal of mmin. Also clamp to the range [0x80,0xff], because
    # 0x100 can't be used as a table entry due to not fitting in a byte, even
    # though it's the exact reciprocal of the overall-smallest mantissa
    # 0x800000.
    gmin = max(128, (2**31 + mmin - 1) // mmax)
    gmax = min(255, 2**31 // mmin)

    # For each of those table entries, compute the result of starting from that
    # value and doing a Newton-Raphson iteration, with the mantissa at each end
    # of the mantissa interval. One of these will be the worst possible error.
    # Choose the table entry whose worst error is as small as possible.
    #
    # (To find the extreme values of a more general function on an interval,
    # you must consider its values not only at the interval endpoints but also
    # any turning points within the interval. Here, the function has only one
    # turning point, and by construction it takes value 0 there, so we needn't
    # worry.)
    g = max(
        range(gmin, gmax + 1),
        key=lambda g: min(
            (g * (2**32 - d * g) / 2**23 - 2**39 / d) for d in [mmin, mmax]
        ),
    )

    print(f"  .byte 0x{g:02x}  // input [0x{mmin:06x},0x{mmax:06x}]"
          f", candidate outputs [0x{gmin:02x},0x{gmax:02x}]"
    )

*/

  .p2align 2  // make sure we start on a 4-byte boundary, even in Thumb
LOCAL_LABEL(tab):
  .byte 0xfe  // input [0x800000,0x81ffff], candidate outputs [0xfd,0xff]
  .byte 0xfa  // input [0x820000,0x83ffff], candidate outputs [0xf9,0xfc]
  .byte 0xf6  // input [0x840000,0x85ffff], candidate outputs [0xf5,0xf8]
  .byte 0xf3  // input [0x860000,0x87ffff], candidate outputs [0xf1,0xf4]
  .byte 0xef  // input [0x880000,0x89ffff], candidate outputs [0xee,0xf0]
  .byte 0xec  // input [0x8a0000,0x8bffff], candidate outputs [0xeb,0xed]
  .byte 0xe8  // input [0x8c0000,0x8dffff], candidate outputs [0xe7,0xea]
  .byte 0xe5  // input [0x8e0000,0x8fffff], candidate outputs [0xe4,0xe6]
  .byte 0xe2  // input [0x900000,0x91ffff], candidate outputs [0xe1,0xe3]
  .byte 0xdf  // input [0x920000,0x93ffff], candidate outputs [0xde,0xe0]
  .byte 0xdc  // input [0x940000,0x95ffff], candidate outputs [0xdb,0xdd]
  .byte 0xd9  // input [0x960000,0x97ffff], candidate outputs [0xd8,0xda]
  .byte 0xd6  // input [0x980000,0x99ffff], candidate outputs [0xd5,0xd7]
  .byte 0xd3  // input [0x9a0000,0x9bffff], candidate outputs [0xd3,0xd4]
  .byte 0xd1  // input [0x9c0000,0x9dffff], candidate outputs [0xd0,0xd2]
  .byte 0xce  // input [0x9e0000,0x9fffff], candidate outputs [0xcd,0xcf]
  .byte 0xcc  // input [0xa00000,0xa1ffff], candidate outputs [0xcb,0xcc]
  .byte 0xc9  // input [0xa20000,0xa3ffff], candidate outputs [0xc8,0xca]
  .byte 0xc7  // input [0xa40000,0xa5ffff], candidate outputs [0xc6,0xc7]
  .byte 0xc4  // input [0xa60000,0xa7ffff], candidate outputs [0xc4,0xc5]
  .byte 0xc2  // input [0xa80000,0xa9ffff], candidate outputs [0xc1,0xc3]
  .byte 0xc0  // input [0xaa0000,0xabffff], candidate outputs [0xbf,0xc0]
  .byte 0xbd  // input [0xac0000,0xadffff], candidate outputs [0xbd,0xbe]
  .byte 0xbb  // input [0xae0000,0xafffff], candidate outputs [0xbb,0xbc]
  .byte 0xb9  // input [0xb00000,0xb1ffff], candidate outputs [0xb9,0xba]
  .byte 0xb7  // input [0xb20000,0xb3ffff], candidate outputs [0xb7,0xb8]
  .byte 0xb5  // input [0xb40000,0xb5ffff], candidate outputs [0xb5,0xb6]
  .byte 0xb3  // input [0xb60000,0xb7ffff], candidate outputs [0xb3,0xb4]
  .byte 0xb1  // input [0xb80000,0xb9ffff], candidate outputs [0xb1,0xb2]
  .byte 0xaf  // input [0xba0000,0xbbffff], candidate outputs [0xaf,0xb0]
  .byte 0xad  // input [0xbc0000,0xbdffff], candidate outputs [0xad,0xae]
  .byte 0xac  // input [0xbe0000,0xbfffff], candidate outputs [0xab,0xac]
  .byte 0xaa  // input [0xc00000,0xc1ffff], candidate outputs [0xa9,0xaa]
  .byte 0xa8  // input [0xc20000,0xc3ffff], candidate outputs [0xa8,0xa8]
  .byte 0xa6  // input [0xc40000,0xc5ffff], candidate outputs [0xa6,0xa7]
  .byte 0xa5  // input [0xc60000,0xc7ffff], candidate outputs [0xa4,0xa5]
  .byte 0xa3  // input [0xc80000,0xc9ffff], candidate outputs [0xa3,0xa3]
  .byte 0xa1  // input [0xca0000,0xcbffff], candidate outputs [0xa1,0xa2]
  .byte 0xa0  // input [0xcc0000,0xcdffff], candidate outputs [0xa0,0xa0]
  .byte 0x9e  // input [0xce0000,0xcfffff], candidate outputs [0x9e,0x9f]
  .byte 0x9d  // input [0xd00000,0xd1ffff], candidate outputs [0x9d,0x9d]
  .byte 0x9b  // input [0xd20000,0xd3ffff], candidate outputs [0x9b,0x9c]
  .byte 0x9a  // input [0xd40000,0xd5ffff], candidate outputs [0x9a,0x9a]
  .byte 0x98  // input [0xd60000,0xd7ffff], candidate outputs [0x98,0x99]
  .byte 0x97  // input [0xd80000,0xd9ffff], candidate outputs [0x97,0x97]
  .byte 0x96  // input [0xda0000,0xdbffff], candidate outputs [0x95,0x96]
  .byte 0x94  // input [0xdc0000,0xddffff], candidate outputs [0x94,0x94]
  .byte 0x93  // input [0xde0000,0xdfffff], candidate outputs [0x93,0x93]
  .byte 0x92  // input [0xe00000,0xe1ffff], candidate outputs [0x91,0x92]
  .byte 0x90  // input [0xe20000,0xe3ffff], candidate outputs [0x90,0x90]
  .byte 0x8f  // input [0xe40000,0xe5ffff], candidate outputs [0x8f,0x8f]
  .byte 0x8e  // input [0xe60000,0xe7ffff], candidate outputs [0x8e,0x8e]
  .byte 0x8d  // input [0xe80000,0xe9ffff], candidate outputs [0x8d,0x8d]
  .byte 0x8b  // input [0xea0000,0xebffff], candidate outputs [0x8b,0x8c]
  .byte 0x8a  // input [0xec0000,0xedffff], candidate outputs [0x8a,0x8a]
  .byte 0x89  // input [0xee0000,0xefffff], candidate outputs [0x89,0x89]
  .byte 0x88  // input [0xf00000,0xf1ffff], candidate outputs [0x88,0x88]
  .byte 0x87  // input [0xf20000,0xf3ffff], candidate outputs [0x87,0x87]
  .byte 0x86  // input [0xf40000,0xf5ffff], candidate outputs [0x86,0x86]
  .byte 0x85  // input [0xf60000,0xf7ffff], candidate outputs [0x85,0x85]
  .byte 0x84  // input [0xf80000,0xf9ffff], candidate outputs [0x84,0x84]
  .byte 0x83  // input [0xfa0000,0xfbffff], candidate outputs [0x83,0x83]
  .byte 0x82  // input [0xfc0000,0xfdffff], candidate outputs [0x82,0x82]
  .byte 0x81  // input [0xfe0000,0xffffff], candidate outputs [0x80,0x81]

END_COMPILERRT_FUNCTION(__aeabi_fdiv)

NO_EXEC_STACK_DIRECTIVE

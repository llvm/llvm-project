//===-- fcmp.h - shared code for single-precision FP comparison functions -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This code is the skeleton of a double-precision FP compare, with two details
// left out: which input value is in which register, and how to make the return
// value. It allows the main comparison logic to be shared between (for
// example) __lesf2 and __gesf2, varying only those details.
//
//===----------------------------------------------------------------------===//

// How to use this header file:
//
// This header file is expected to be #included from inside a function
// definition in a .S file. The source file including this header should
// provide the following:
//
// op0 and op1: register aliases (via .req) for the registers containing the
// input operands.
//  - For most comparisons, op0 will correspond to r0 and op1 to r1.
//  - But a function with the reversed semantics of __aeabi_cfrcmple wil define
//    them the other way round.
//
// SetReturnRegister: an assembly macro that looks at the PSR flags and sets up
// an appropriate return value in r0, for the cases that do *not* involve NaN.
//  - On entry to this macro, the condition codes LO, EQ and HI indicate that
//    op0 < op1, op0 == op1 or op0 > op1 respectively.
//  - For functions that return a result in the flags, this macro can be empty,
//    because those are the correct flags to return anyway.
//  - Functions that return a boolean in r0 should set it up by checking the
//    flags.
//
// LOCAL_LABEL(NaN): a label defined within the compare function, after the
// #include of this header. Called when at least one input is a NaN, and sets
// up the appropriate return value for that case.

// --------------------------------------------------
// The actual entry point of the compare function.
//
// The basic plan is to start by ORing together the two inputs. This tells us
// two things:
//  - the top bit of the output tells us whether both inputs are positive, or
//    whether at least one is negative
//  - if the 8 exponent bits of the output are not all 1, then there are
//    definitely no NaNs, so a fast path can handle most non-NaN cases.

  // Set up the constant 1 << 23 in a register, which we'll need on all
  // branches.
  movs    r3, #1
  lsls    r3, r3, #23

  // Diverge control for the negative-numbers case.
  movs    r2, op0
  orrs    r2, r2, op1
  bmi     LOCAL_LABEL(negative)         // high bit set => at least one negative input

  // Here, both inputs are positive. Try adding 1<<23 to their bitwise OR in
  // r2. This will carry all the way into the top bit, setting the N flag, if
  // all 8 exponent bits were set.
  cmn     r2, r3
  bmi     LOCAL_LABEL(NaNInf_check_positive) // need to look harder for NaNs

  // The fastest fast path: both inputs positive and we could easily tell there
  // were no NaNs. So we just compare op0 and op1 as unsigned integers.
  cmp     op0, op1
  SetReturnRegister
  bx      lr

LOCAL_LABEL(NaNInf_check_positive):
  // Second tier for positive numbers. We come here if both inputs are
  // positive, but our fast initial check didn't manage to rule out a NaN. But
  // it's not guaranteed that there _is_ a NaN, for two reasons:
  //
  //  1. An input with exponent 0xFF might be an infinity instead. Those behave
  //    normally under comparison.
  //
  //  2. There might not even _be_ an input with exponent 0xFF. All we know so
  //     far is that the two inputs ORed together had all the exponent bits
  //     set. So each of those bits is set in _at least one_ of the inputs, but
  //     not necessarily all in the _same_ input.
  //
  // Test each exponent individually for 0xFF, using the same CMN idiom as
  // above. If neither one carries into the sign bit then we have no NaNs _or_
  // infinities and can compare the registers and return again.
  cmn     op0, r3
  bmi     LOCAL_LABEL(NaN_check_positive)
  cmn     op1, r3
  bmi     LOCAL_LABEL(NaN_check_positive)

  // Second-tier return path, now we've ruled out anything difficult.
  cmp     op0, op1
  SetReturnRegister
  bx      lr

LOCAL_LABEL(NaN_check_positive):
  // Third tier for positive numbers. Here we know that at least one of the
  // inputs has exponent 0xFF. But they might still be infinities rather than
  // NaNs. So now we must check whether there's an actual NaN, by shifting each
  // input left to get rid of the sign bit, and seeing if the result is
  // _greater_ than 0xFF000000 (but not equal).
  //
  // We could have skipped the second-tier check and done this more rigorous
  // test immediately. But that would cost an extra instruction in the case
  // where there are no infinities or NaNs, and we assume that that is so much
  // more common that it's worth optimizing for.
  movs    r2, #0xFF
  lsls    r2, r2, #24
  lsls    r3, op0, #1
  cmp     r3, r2
  bhi     LOCAL_LABEL(NaN)
  lsls    r3, op1, #1
  cmp     r3, r2
  bhi     LOCAL_LABEL(NaN)

  // Now we've finally ruled out NaNs! And we still know both inputs are
  // positive. So the third-tier return path can just compare the numbers
  // again.
  cmp     op0, op1
  SetReturnRegister
  bx      lr

LOCAL_LABEL(negative):
  // We come here if at least one operand is negative. We haven't checked for
  // NaNs at all yet (the sign check came first), so repeat the first-tier
  // check strategy of seeing if all exponent bits are set in r12.
  //
  // On this path, the sign bit in r12 is set, so if adding 1 to the low
  // exponent bit carries all the way through into the sign bit, it will
  // _clear_ the sign bit rather than setting it. So we expect MI to be the
  // "definitely no NaNs" result, where it was PL on the positive branch.
  cmn     r2, r3
  bpl     LOCAL_LABEL(NaNInf_check_negative)

  // Now we have no NaNs, but at least one negative number. This gives us two
  // complications:
  //
  //  1. Floating-point numbers are sign/magnitude, not two's complement, so we
  //     have to consider separately the cases of "both negative" and "one of
  //     each sign".
  //
  //  2. -0 and +0 are required to compare equal.
  //
  // But problem #1 is not as hard as it sounds! If both operands are negative,
  // then we can get the result we want by comparing them as unsigned integers
  // the opposite way round, because the input with the smaller value (as an
  // integer) is the larger number in an FP ordering sense. And if one operand
  // is negative and the other is positive, the _same_ reversed comparison
  // works, because the positive number (with zero sign bit) will always
  // compare less than the negative one in an unsigned-integers sense.
  //
  // So we only have to worry about problem #2, signed zeroes. This only
  // affects the answer if _both_ operands are zero. And we can check that
  // easily, because it happens if and only if r12 = 0x80000000. (We know r12
  // has its sign bit set; if it has no other bits set, that's because both
  // inputs were either 0x80000000 or 0x00000000.)
  lsls    r2, r2, #1              // EQ if both inputs are zero (also sets C)
  beq     1f
  cmp     op1, op0                // otherwise, compare them backwards
1:
  SetReturnRegister
  bx      lr

LOCAL_LABEL(NaNInf_check_negative):
  // Second tier for negative numbers: we know the OR of the exponents is 0xFF,
  // but again, we might not have either _actual_ exponent 0xFF, and also, an
  // exponent 0xFF might be an infinity instead of a NaN.
  //
  // On this path we've already branched twice (once for negative numbers and
  // once for the first-tier NaN check), so we'll just go straight to the
  // precise check for NaNs.
  movs    r2, #0xFF
  lsls    r2, r2, #24
  lsls    r3, op0, #1
  cmp     r3, r2
  bhi     LOCAL_LABEL(NaN)
  lsls    r3, op1, #1
  cmp     r3, r2
  bhi     LOCAL_LABEL(NaN)

  // Now we've ruled out NaNs, so we can just compare the two input registers
  // and return. On this path we _don't_ need to check for the special case of
  // comparing two zeroes, because we only came here if the bitwise OR of the
  // exponent fields was 0xFF, which means the exponents can't both have been
  // zero! So we can _just_ do the reversed CMP and finish.
  cmp     op1, op0
  SetReturnRegister
  bx      lr

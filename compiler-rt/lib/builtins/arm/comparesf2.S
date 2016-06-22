//===-- comparesf2.S - Implement single-precision soft-float comparisons --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the following soft-fp_t comparison routines:
//
//   __eqsf2   __gesf2   __unordsf2
//   __lesf2   __gtsf2
//   __ltsf2
//   __nesf2
//
// The semantics of the routines grouped in each column are identical, so there
// is a single implementation for each, with multiple names.
//
// The routines behave as follows:
//
//   __lesf2(a,b) returns -1 if a < b
//                         0 if a == b
//                         1 if a > b
//                         1 if either a or b is NaN
//
//   __gesf2(a,b) returns -1 if a < b
//                         0 if a == b
//                         1 if a > b
//                        -1 if either a or b is NaN
//
//   __unordsf2(a,b) returns 0 if both a and b are numbers
//                           1 if either a or b is NaN
//
// Note that __lesf2( ) and __gesf2( ) are identical except in their handling of
// NaN values.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"
.syntax unified

.p2align 2
DEFINE_COMPILERRT_FUNCTION(__eqsf2)
    // Make copies of a and b with the sign bit shifted off the top.  These will
    // be used to detect zeros and NaNs.
    mov     r2,         r0, lsl #1
    mov     r3,         r1, lsl #1

    // We do the comparison in three stages (ignoring NaN values for the time
    // being).  First, we orr the absolute values of a and b; this sets the Z
    // flag if both a and b are zero (of either sign).  The shift of r3 doesn't
    // effect this at all, but it *does* make sure that the C flag is clear for
    // the subsequent operations.
    orrs    r12,    r2, r3, lsr #1

    // Next, we check if a and b have the same or different signs.  If they have
    // opposite signs, this eor will set the N flag.
    it ne
    eorsne  r12,    r0, r1

    // If a and b are equal (either both zeros or bit identical; again, we're
    // ignoring NaNs for now), this subtract will zero out r0.  If they have the
    // same sign, the flags are updated as they would be for a comparison of the
    // absolute values of a and b.
    it pl
    subspl  r0,     r2, r3

    // If a is smaller in magnitude than b and both have the same sign, place
    // the negation of the sign of b in r0.  Thus, if both are negative and
    // a > b, this sets r0 to 0; if both are positive and a < b, this sets
    // r0 to -1.
    //
    // This is also done if a and b have opposite signs and are not both zero,
    // because in that case the subtract was not performed and the C flag is
    // still clear from the shift argument in orrs; if a is positive and b
    // negative, this places 0 in r0; if a is negative and b positive, -1 is
    // placed in r0.
    it lo
    mvnlo   r0,         r1, asr #31

    // If a is greater in magnitude than b and both have the same sign, place
    // the sign of b in r0.  Thus, if both are negative and a < b, -1 is placed
    // in r0, which is the desired result.  Conversely, if both are positive
    // and a > b, zero is placed in r0.
    it hi
    movhi   r0,         r1, asr #31

    // If you've been keeping track, at this point r0 contains -1 if a < b and
    // 0 if a >= b.  All that remains to be done is to set it to 1 if a > b.
    // If a == b, then the Z flag is set, so we can get the correct final value
    // into r0 by simply or'ing with 1 if Z is clear.
    it ne
    orrne   r0,     r0, #1

    // Finally, we need to deal with NaNs.  If either argument is NaN, replace
    // the value in r0 with 1.
    cmp     r2,         #0xff000000
    ite ls
    cmpls   r3,         #0xff000000
    movhi   r0,         #1
    JMP(lr)
END_COMPILERRT_FUNCTION(__eqsf2)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__lesf2, __eqsf2)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__ltsf2, __eqsf2)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__nesf2, __eqsf2)

.p2align 2
DEFINE_COMPILERRT_FUNCTION(__gtsf2)
    // Identical to the preceding except in that we return -1 for NaN values.
    // Given that the two paths share so much code, one might be tempted to 
    // unify them; however, the extra code needed to do so makes the code size
    // to performance tradeoff very hard to justify for such small functions.
    mov     r2,         r0, lsl #1
    mov     r3,         r1, lsl #1
    orrs    r12,    r2, r3, lsr #1
    it ne
    eorsne  r12,    r0, r1
    it pl
    subspl  r0,     r2, r3
    it lo
    mvnlo   r0,         r1, asr #31
    it hi
    movhi   r0,         r1, asr #31
    it ne
    orrne   r0,     r0, #1
    cmp     r2,         #0xff000000
    ite ls
    cmpls   r3,         #0xff000000
    movhi   r0,         #-1
    JMP(lr)
END_COMPILERRT_FUNCTION(__gtsf2)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__gesf2, __gtsf2)

.p2align 2
DEFINE_COMPILERRT_FUNCTION(__unordsf2)
    // Return 1 for NaN values, 0 otherwise.
    mov     r2,         r0, lsl #1
    mov     r3,         r1, lsl #1
    mov     r0,         #0
    cmp     r2,         #0xff000000
    ite ls
    cmpls   r3,         #0xff000000
    movhi   r0,         #1
    JMP(lr)
END_COMPILERRT_FUNCTION(__unordsf2)

DEFINE_AEABI_FUNCTION_ALIAS(__aeabi_fcmpun, __unordsf2)

NO_EXEC_STACK_DIRECTIVE


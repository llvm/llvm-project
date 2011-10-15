/*===-- divmodsi4.S - 32-bit signed integer divide and modulus ------------===//
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===//
 *
 * This file implements the __divmodsi4 (32-bit signed integer divide and
 * modulus) function for the ARM architecture.  A naive digit-by-digit
 * computation is employed for simplicity.
 *
 *===----------------------------------------------------------------------===*/

#include "../assembly.h"

#define ESTABLISH_FRAME    \
    push   {r4-r7, lr}   ;\
    add     r7,     sp, #12
#define CLEAR_FRAME_AND_RETURN \
    pop    {r4-r7, pc}

.syntax unified
.align 3
DEFINE_COMPILERRT_FUNCTION(__divmodsi4)
    ESTABLISH_FRAME
//  Set aside the sign of the quotient and modulus, and the address for the
//  modulus.
    eor     r4,     r0, r1
    mov     r5,     r0
    mov     r6,     r2
//  Take the absolute value of a and b via abs(x) = (x^(x >> 31)) - (x >> 31).
    eor     ip,     r0, r0, asr #31
    eor     lr,     r1, r1, asr #31
    sub     r0,     ip, r0, asr #31
    sub     r1,     lr, r1, asr #31
//  Unsigned divmod:
    bl      SYMBOL_NAME(__udivmodsi4)
//  Apply the sign of quotient and modulus
    ldr     r1,    [r6]
    eor     r0,     r0, r4, asr #31
    eor     r1,     r1, r5, asr #31
    sub     r0,     r0, r4, asr #31
    sub     r1,     r1, r5, asr #31
    str     r1,    [r6]
    CLEAR_FRAME_AND_RETURN

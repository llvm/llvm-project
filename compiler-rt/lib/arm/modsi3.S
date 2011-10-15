/*===-- modsi3.S - 32-bit signed integer modulus --------------------------===//
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===//
 *
 * This file implements the __modsi3 (32-bit signed integer modulus) function
 * for the ARM architecture as a wrapper around the unsigned routine.
 *
 *===----------------------------------------------------------------------===*/

#include "../assembly.h"

#define ESTABLISH_FRAME \
    push   {r4, r7, lr}    ;\
    add     r7,     sp, #4
#define CLEAR_FRAME_AND_RETURN \
    pop    {r4, r7, pc}

.syntax unified
.align 3
DEFINE_COMPILERRT_FUNCTION(__modsi3)
    ESTABLISH_FRAME
    //  Set aside the sign of the dividend.
    mov     r4,     r0
    //  Take absolute value of a and b via abs(x) = (x^(x >> 31)) - (x >> 31).
    eor     r2,     r0, r0, asr #31
    eor     r3,     r1, r1, asr #31
    sub     r0,     r2, r0, asr #31
    sub     r1,     r3, r1, asr #31
    //  abs(a) % abs(b)
    bl     SYMBOL_NAME(__umodsi3)
    //  Apply sign of dividend to result and return.
    eor     r0,     r0, r4, asr #31
    sub     r0,     r0, r4, asr #31
    CLEAR_FRAME_AND_RETURN

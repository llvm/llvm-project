/*===-- udivsi3.S - 32-bit unsigned integer divide ------------------------===//
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===//
 *
 * This file implements the __udivsi3 (32-bit unsigned integer divide) 
 * function for the ARM architecture.  A naive digit-by-digit computation is
 * employed for simplicity.
 *
 *===----------------------------------------------------------------------===*/

#include "../assembly.h"

#define ESTABLISH_FRAME \
    push   {r7, lr}    ;\
    mov     r7,     sp
#define CLEAR_FRAME_AND_RETURN \
    pop    {r7, pc}

#define a r0
#define b r1
#define r r2
#define i r3
#define q ip
#define one lr

.syntax unified
.align 3
// Ok, APCS and AAPCS agree on 32 bit args, so it's safe to use the same routine.
DEFINE_AEABI_FUNCTION_ALIAS(__aeabi_uidiv, __udivsi3)
DEFINE_COMPILERRT_FUNCTION(__udivsi3)
//  We use a simple digit by digit algorithm; before we get into the actual 
//  divide loop, we must calculate the left-shift amount necessary to align
//  the MSB of the divisor with that of the dividend (If this shift is
//  negative, then the result is zero, and we early out). We also conjure a
//  bit mask of 1 to use in constructing the quotient, and initialize the
//  quotient to zero.
    ESTABLISH_FRAME
    clz     r2,     a
    tst     b,      b   // detect divide-by-zero
    clz     r3,     b
    mov     q,      #0
    beq     LOCAL_LABEL(return)    // return 0 if b is zero.
    mov     one,    #1
    subs    i,      r3, r2
    blt     LOCAL_LABEL(return)    // return 0 if MSB(a) < MSB(b)

LOCAL_LABEL(mainLoop):
//  This loop basically implements the following:
//
//  do {
//      if (a >= b << i) {
//          a -= b << i;
//          q |= 1 << i;
//          if (a == 0) break;
//      }
//  } while (--i)
//
//  Note that this does not perform the final iteration (i == 0); by doing it
//  this way, we can merge the two branches which is a substantial win for
//  such a tight loop on current ARM architectures.
    subs    r,      a,  b, lsl i
    orrhs   q,      q,one, lsl i
    movhs   a,      r
    subsne  i,      i, #1
    bhi     LOCAL_LABEL(mainLoop)

//  Do the final test subtraction and update of quotient (i == 0), as it is
//  not performed in the main loop.
    subs    r,      a,  b
    orrhs   q,      #1

LOCAL_LABEL(return):
//  Move the quotient to r0 and return.
    mov     r0,     q
    CLEAR_FRAME_AND_RETURN

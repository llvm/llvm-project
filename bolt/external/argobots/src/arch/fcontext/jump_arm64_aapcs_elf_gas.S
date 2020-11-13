/*
            Copyright Edward Nevill 2015
   Distributed under the Boost Software License, Version 1.0.
      (See accompanying file LICENSE_1_0.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
/*******************************************************
 *                                                     *
 *  -------------------------------------------------  *
 *  |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  *
 *  -------------------------------------------------  *
 *  | 0x0 | 0x4 | 0x8 | 0xc | 0x10| 0x14| 0x18| 0x1c|  *
 *  -------------------------------------------------  *
 *  |    d8     |    d9     |    d10    |    d11    |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  8  |  9  |  10 |  11 |  12 |  13 |  14 |  15 |  *
 *  -------------------------------------------------  *
 *  | 0x20| 0x24| 0x28| 0x2c| 0x30| 0x34| 0x38| 0x3c|  *
 *  -------------------------------------------------  *
 *  |    d12    |    d13    |    d14    |    d15    |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  16 |  17 |  18 |  19 |  20 |  21 |  22 |  23 |  *
 *  -------------------------------------------------  *
 *  | 0x40| 0x44| 0x48| 0x4c| 0x50| 0x54| 0x58| 0x5c|  *
 *  -------------------------------------------------  *
 *  |    x19    |    x20    |    x21    |    x22    |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  24 |  25 |  26 |  27 |  28 |  29 |  30 |  31 |  *
 *  -------------------------------------------------  *
 *  | 0x60| 0x64| 0x68| 0x6c| 0x70| 0x74| 0x78| 0x7c|  *
 *  -------------------------------------------------  *
 *  |    x23    |    x24    |    x25    |    x26    |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  32 |  33 |  34 |  35 |  36 |  37 |  38 |  39 |  *
 *  -------------------------------------------------  *
 *  | 0x80| 0x84| 0x88| 0x8c| 0x90| 0x94| 0x98| 0x9c|  *
 *  -------------------------------------------------  *
 *  |    x27    |    x28    |    FP     |     LR    |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  40 |  41 |  42 | 43  |           |           |  *
 *  -------------------------------------------------  *
 *  | 0xa0| 0xa4| 0xa8| 0xac|           |           |  *
 *  -------------------------------------------------  *
 *  |     PC    |   align   |           |           |  *
 *  -------------------------------------------------  *
 *                                                     *
 *******************************************************/

#include "abt_config.h"

.cpu    generic+fp+simd
.text
.align  2
.global jump_fcontext
.type   jump_fcontext, %function
jump_fcontext:
    # prepare stack for GP + FPU
    sub  sp, sp, #0xb0

#if ABTD_FCONTEXT_PRESERVE_FPU
    # save d8 - d15
    stp  d8,  d9,  [sp, #0x00]
    stp  d10, d11, [sp, #0x10]
    stp  d12, d13, [sp, #0x20]
    stp  d14, d15, [sp, #0x30]
#endif

    # save x19-x30
    stp  x19, x20, [sp, #0x40]
    stp  x21, x22, [sp, #0x50]
    stp  x23, x24, [sp, #0x60]
    stp  x25, x26, [sp, #0x70]
    stp  x27, x28, [sp, #0x80]
    stp  x29, x30, [sp, #0x90]

    # save LR as PC
    str  x30, [sp, #0xa0]

    # store RSP (pointing to context-data) in first argument (x0).
    # STR cannot have sp as a target register
    mov  x4, sp
    str  x4, [x0]

    # restore RSP (pointing to context-data) from A2 (x1)
    mov  sp, x1

#if ABTD_FCONTEXT_PRESERVE_FPU
    # load d8 - d15
    ldp  d8,  d9,  [sp, #0x00]
    ldp  d10, d11, [sp, #0x10]
    ldp  d12, d13, [sp, #0x20]
    ldp  d14, d15, [sp, #0x30]
#endif

    # load x19-x30
    ldp  x19, x20, [sp, #0x40]
    ldp  x21, x22, [sp, #0x50]
    ldp  x23, x24, [sp, #0x60]
    ldp  x25, x26, [sp, #0x70]
    ldp  x27, x28, [sp, #0x80]
    ldp  x29, x30, [sp, #0x90]

    # use third arg as return value after jump
    # and as first arg in context function
    mov  x0, x2

    # load pc
    ldr  x4, [sp, #0xa0]

    # restore stack from GP + FPU
    add  sp, sp, #0xb0

    ret x4
.size   jump_fcontext,.-jump_fcontext

#if ABT_CONFIG_THREAD_TYPE == ABT_THREAD_TYPE_DYNAMIC_PROMOTION
.text
.globl init_and_call_fcontext
.type init_and_call_fcontext,@function
.align 16
init_and_call_fcontext:
    # save the current rsp to [sp - 0x08]
    mov x4, sp
    stp x30, x4, [x2, -0x10]

    # save callee-saved registers
    # prepare stack for GP + FPU
    sub  sp, sp, #0xb0

#if ABTD_FCONTEXT_PRESERVE_FPU
    # save d8 - d15
    stp  d8,  d9,  [sp, #0x00]
    stp  d10, d11, [sp, #0x10]
    stp  d12, d13, [sp, #0x20]
    stp  d14, d15, [sp, #0x30]
#endif

    # save x19-x30
    stp  x19, x20, [sp, #0x40]
    stp  x21, x22, [sp, #0x50]
    stp  x23, x24, [sp, #0x60]
    stp  x25, x26, [sp, #0x70]
    stp  x27, x28, [sp, #0x80]
    stp  x29, x30, [sp, #0x90]

    # save LR as PC
    str  x30, [sp, #0xa0]

    # store sp in x3 (= fctx)
    mov x4, sp
    str x4, [x3]

    # assign x2 (=p_stack) to sp
    sub sp, x2, #0x10

    # call x1 (= f_thread). x0 (= p_arg) has been already set
    # sp is 16-byte aligned (ABI specification)
    blr x1

    # restore original LR and sp
    ldp x30, x3, [sp]
    mov sp, x3

    # - when the thread did not yield, sp is the original one, so ret jumps to
    #   the original control flow.
    # - any suspension updates sp to (p_stack - 0x10). so that ret calls
    #   (p_stack - 0x8), which is a set to a terminator function
    # sp is 16-byte aligned (ABI specification)
    ret
.size init_and_call_fcontext,.-init_and_call_fcontext
#endif

# Mark that we don't need executable stack.
.section .note.GNU-stack,"",%progbits


/*
            Copyright Oliver Kowalke 2009.
   Distributed under the Boost Software License, Version 1.0.
      (See accompanying file LICENSE_1_0.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/

/*******************************************************
 *                                                     *
 *  -------------------------------------------------  *
 *  |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  *
 *  -------------------------------------------------  *
 *  |  0  |  4  |  8  |  12 |  16 |  20 |  24 |  28 |  *
 *  -------------------------------------------------  *
 *  |    F14    |    F15    |    F16    |    F17    |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  8  |  9  |  10 |  11 |  12 |  13 |  14 |  15 |  *
 *  -------------------------------------------------  *
 *  |  32 |  36 |  40 |  44 |  48 |  52 |  56 |  60 |  *
 *  -------------------------------------------------  *
 *  |    F18    |    F19    |    F20    |    F21    |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  16 |  17 |  18 |  19 |  20 |  21 |  22 |  23 |  *
 *  -------------------------------------------------  *
 *  |  64 |  68 |  72 |  76 |  80 |  84 |  88 |  92 |  *
 *  -------------------------------------------------  *
 *  |    F22    |    F23    |    F24    |    F25    |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  24 |  25 |  26 |  27 |  28 |  29 |  30 |  31 |  *
 *  -------------------------------------------------  *
 *  |  96 | 100 | 104 | 108 | 112 | 116 | 120 | 124 |  *
 *  -------------------------------------------------  *
 *  |    F26    |    F27    |    F28    |    F29    |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  32 |  33 |  34 |  35 |  36 |  37 |  38 |  39 |  *
 *  -------------------------------------------------  *
 *  | 128 | 132 | 136 | 140 | 144 | 148 | 152 | 156 |  *
 *  -------------------------------------------------  *
 *  |    F30    |    F31    |   fpscr   |    vscr   |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  40 |  41 |  42 |  43 |  44 |  45 |  46 |  47 |  *
 *  -------------------------------------------------  *
 *  | 160 | 164 | 168 | 172 | 176 | 180 | 184 | 188 |  *
 *  -------------------------------------------------  *
 *  |          V20          |          V21          |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  48 |  49 |  50 |  51 |  52 |  53 |  54 |  55 |  *
 *  -------------------------------------------------  *
 *  | 192 | 196 | 200 | 204 | 208 | 212 | 216 | 220 |  *
 *  -------------------------------------------------  *
 *  |          V22          |          V23          |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  56 |  57 |  58 |  59 |  60 |  61 |  62 |  63 |  *
 *  -------------------------------------------------  *
 *  | 224 | 228 | 232 | 236 | 240 | 244 | 248 | 252 |  *
 *  -------------------------------------------------  *
 *  |          V24          |          V25          |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  64 |  65 |  66 |  67 |  68 |  69 |  70 |  71 |  *
 *  -------------------------------------------------  *
 *  | 256 | 260 | 264 | 268 | 272 | 276 | 280 | 284 |  *
 *  -------------------------------------------------  *
 *  |          V26          |          V27          |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  72 |  73 |  74 |  75 |  76 |  77 |  78 |  79 |  *
 *  -------------------------------------------------  *
 *  | 288 | 292 | 296 | 300 | 304 | 308 | 312 | 316 |  *
 *  -------------------------------------------------  *
 *  |          V28          |          V29          |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  80 |  81 |  82 |  83 |  84 |  85 |  86 |  87 |  *
 *  -------------------------------------------------  *
 *  | 320 | 324 | 328 | 332 | 336 | 340 | 344 | 348 |  *
 *  -------------------------------------------------  *
 *  |          V30          |          V31          |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  88 |  89 |  90 |  91 |  92 |  93 |  94 |  95 |  *
 *  -------------------------------------------------  *
 *  | 352 | 356 | 360 | 364 | 368 | 372 | 376 | 380 |  *
 *  -------------------------------------------------  *
 *  |    R14    |    R15    |     R16   |    R17    |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |  96 |  97 |  98 |  99 | 100 | 101 | 102 | 103 |  *
 *  -------------------------------------------------  *
 *  | 384 | 388 | 392 | 396 | 400 | 404 | 408 | 412 |  *
 *  -------------------------------------------------  *
 *  |    R18    |    R19    |    R20    |    R21    |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 |  *
 *  -------------------------------------------------  *
 *  | 416 | 420 | 424 | 428 | 432 | 436 | 440 | 444 |  *
 *  -------------------------------------------------  *
 *  |    R22    |    R23    |    R24    |    R25    |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 |  *
 *  -------------------------------------------------  *
 *  | 448 | 452 | 456 | 460 | 464 | 468 | 472 | 476 |  *
 *  -------------------------------------------------  *
 *  |    R26    |    R27    |    R28    |    R29    |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  | 120 | 121 | 122 | 123 | 124 | 125 | 126 | 127 |  *
 *  -------------------------------------------------  *
 *  | 480 | 484 | 488 | 492 | 496 | 500 | 504 | 508 |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  |    R30    |    R31    |    TOC    |     CR    |  *
 *  -------------------------------------------------  *
 *  -------------------------------------------------  *
 *  | 128 | 129 | 130 | 131 |                       |  *
 *  -------------------------------------------------  *
 *  | 512 | 516 | 520 | 524 |                       |  *
 *  -------------------------------------------------  *
 *  |    LR     |    PC     |                       |  *
 *  -------------------------------------------------  *
 *                                                     *
 *******************************************************/

#include "abt_config.h"

.globl jump_fcontext
#if _CALL_ELF == 2
	.text
	.align 2
jump_fcontext:
        addis   %r2, %r12, .TOC.-jump_fcontext@ha
        addi    %r2, %r2, .TOC.-jump_fcontext@l
        .localentry jump_fcontext, . - jump_fcontext
#else
	.section ".opd","aw"
	.align 3
jump_fcontext:
# ifdef _CALL_LINUX
        .quad   .L.jump_fcontext,.TOC.@tocbase,0
        .type   jump_fcontext,@function
        .text
        .align 2
.L.jump_fcontext:
# else
        .hidden .jump_fcontext
        .globl  .jump_fcontext
        .quad   .jump_fcontext,.TOC.@tocbase,0
        .size   jump_fcontext,24
        .type   .jump_fcontext,@function
        .text
        .align 2
.jump_fcontext:
# endif
#endif
    # reserve space on stack
    subi  %r1, %r1, 528

    std  %r14, 352(%r1)  # save R14
    std  %r15, 360(%r1)  # save R15
    std  %r16, 368(%r1)  # save R16
    std  %r17, 376(%r1)  # save R17
    std  %r18, 384(%r1)  # save R18
    std  %r19, 392(%r1)  # save R19
    std  %r20, 400(%r1)  # save R20
    std  %r21, 408(%r1)  # save R21
    std  %r22, 416(%r1)  # save R22
    std  %r23, 424(%r1)  # save R23
    std  %r24, 432(%r1)  # save R24
    std  %r25, 440(%r1)  # save R25
    std  %r26, 448(%r1)  # save R26
    std  %r27, 456(%r1)  # save R27
    std  %r28, 464(%r1)  # save R28
    std  %r29, 472(%r1)  # save R29
    std  %r30, 480(%r1)  # save R30
    std  %r31, 488(%r1)  # save R31
#if _CALL_ELF != 2
    std  %r2,  496(%r1)  # save TOC
#endif

    # save CR
    mfcr  %r0
    std  %r0, 504(%r1)
    # save LR
    mflr  %r0
    std  %r0, 512(%r1)
    # save LR as PC
    std  %r0, 520(%r1)

#if ABTD_FCONTEXT_PRESERVE_FPU
    stfd  %f14, 0(%r1)  # save F14
    stfd  %f15, 8(%r1)  # save F15
    stfd  %f16, 16(%r1)  # save F16
    stfd  %f17, 24(%r1)  # save F17
    stfd  %f18, 32(%r1)  # save F18
    stfd  %f19, 40(%r1)  # save F19
    stfd  %f20, 48(%r1)  # save F20
    stfd  %f21, 56(%r1)  # save F21
    stfd  %f22, 64(%r1)  # save F22
    stfd  %f23, 72(%r1)  # save F23
    stfd  %f24, 80(%r1)  # save F24
    stfd  %f25, 88(%r1)  # save F25
    stfd  %f26, 96(%r1)  # save F26
    stfd  %f27, 104(%r1)  # save F27
    stfd  %f28, 112(%r1)  # save F28
    stfd  %f29, 120(%r1)  # save F29
    stfd  %f30, 128(%r1)  # save F30
    stfd  %f31, 136(%r1)  # save F31
#ifdef __VSX__
    # VSCR can be loaded only to Vn.  To store VSCR as it is a vector, it must
    # be written before saving FPSCR.
    mfvscr %v19           # load VSCR
    li    %r10, 144
    stvx  %v19, %r10, %r1 # save VSCR.  Only the last 32 bits are used
#endif
    mffs  %f0  # load FPSCR
    stfd  %f0, 144(%r1)  # save FPSCR
#ifdef __VSX__
    # OpenPOWER saves V20-V31 (vector units)
    # note stvx cannot take an immediate value as an offset.
    li    %r14, 160
    stvx  %v20, %r14, %r1  # save V20
    li    %r15, 176
    stvx  %v21, %r15, %r1  # save V21
    li    %r16, 192
    stvx  %v22, %r16, %r1  # save V22
    li    %r17, 208
    stvx  %v23, %r17, %r1  # save V23
    li    %r18, 224
    stvx  %v24, %r18, %r1  # save V24
    li    %r19, 240
    stvx  %v25, %r19, %r1  # save V25
    li    %r20, 256
    stvx  %v26, %r20, %r1  # save V26
    li    %r21, 272
    stvx  %v27, %r21, %r1  # save V27
    li    %r22, 288
    stvx  %v28, %r22, %r1  # save V28
    li    %r23, 304
    stvx  %v29, %r23, %r1  # save V29
    li    %r24, 320
    stvx  %v30, %r24, %r1  # save V30
    li    %r25, 336
    stvx  %v31, %r25, %r1  # save V31
#endif
#endif

    # store RSP (pointing to context-data) in R3
    std  %r1, 0(%r3)

    # restore RSP (pointing to context-data) from R4
    mr  %r1, %r4

#if ABTD_FCONTEXT_PRESERVE_FPU
    lfd  %f14, 0(%r1)  # restore F14
    lfd  %f15, 8(%r1)  # restore F15
    lfd  %f16, 16(%r1)  # restore F16
    lfd  %f17, 24(%r1)  # restore F17
    lfd  %f18, 32(%r1)  # restore F18
    lfd  %f19, 40(%r1)  # restore F19
    lfd  %f20, 48(%r1)  # restore F20
    lfd  %f21, 56(%r1)  # restore F21
    lfd  %f22, 64(%r1)  # restore F22
    lfd  %f23, 72(%r1)  # restore F23
    lfd  %f24, 80(%r1)  # restore F24
    lfd  %f25, 88(%r1)  # restore F25
    lfd  %f26, 96(%r1)  # restore F26
    lfd  %f27, 104(%r1)  # restore F27
    lfd  %f28, 112(%r1)  # restore F28
    lfd  %f29, 120(%r1)  # restore F29
    lfd  %f30, 128(%r1)  # restore F30
    lfd  %f31, 136(%r1)  # restore F31
    lfd  %f0,  144(%r1)  # load FPSCR
    mtfsf  0xff, %f0  # restore FPSCR
#ifdef __VSX__
    li    %r10, 144
    lvx   %v19, %r10, %r1 # load VSCR
    mtvscr %v19           # restore VSCR.  Only the last 32 bits are used
    li    %r14, 160
    lvx  %v20, %r14, %r1  # restore V20
    li    %r15, 176
    lvx  %v21, %r15, %r1  # restore V21
    li    %r16, 192
    lvx  %v22, %r16, %r1  # restore V22
    li    %r17, 208
    lvx  %v23, %r17, %r1  # restore V23
    li    %r18, 224
    lvx  %v24, %r18, %r1  # restore V24
    li    %r19, 240
    lvx  %v25, %r19, %r1  # restore V25
    li    %r20, 256
    lvx  %v26, %r20, %r1  # restore V26
    li    %r21, 272
    lvx  %v27, %r21, %r1  # restore V27
    li    %r22, 288
    lvx  %v28, %r22, %r1  # restore V28
    li    %r23, 304
    lvx  %v29, %r23, %r1  # restore V29
    li    %r24, 320
    lvx  %v30, %r24, %r1  # restore V30
    li    %r25, 336
    lvx  %v31, %r25, %r1  # restore V31
#endif
#endif

    ld  %r14, 352(%r1)  # restore R14
    ld  %r15, 360(%r1)  # restore R15
    ld  %r16, 368(%r1)  # restore R16
    ld  %r17, 376(%r1)  # restore R17
    ld  %r18, 384(%r1)  # restore R18
    ld  %r19, 392(%r1)  # restore R19
    ld  %r20, 400(%r1)  # restore R20
    ld  %r21, 408(%r1)  # restore R21
    ld  %r22, 416(%r1)  # restore R22
    ld  %r23, 424(%r1)  # restore R23
    ld  %r24, 432(%r1)  # restore R24
    ld  %r25, 440(%r1)  # restore R25
    ld  %r26, 448(%r1)  # restore R26
    ld  %r27, 456(%r1)  # restore R27
    ld  %r28, 464(%r1)  # restore R28
    ld  %r29, 472(%r1)  # restore R29
    ld  %r30, 480(%r1)  # restore R30
    ld  %r31, 488(%r1)  # restore R31
#if _CALL_ELF != 2
    ld  %r2,  496(%r1)  # restore TOC
#endif

    # restore CR
    ld  %r0, 504(%r1)
    mtcr  %r0
    # restore LR
    ld  %r0, 512(%r1)
    mtlr  %r0

    # load PC
    ld  %r12, 520(%r1)
    # restore CTR
    mtctr  %r12

    # adjust stack
    addi  %r1, %r1, 528

    # use third arg as return value after jump
    # use third arg as first arg in context function
    mr  %r3, %r5

    # jump to context
    bctr
#if _CALL_ELF == 2
	.size jump_fcontext, .-jump_fcontext
#else
# ifdef _CALL_LINUX
	.size .jump_fcontext, .-.L.jump_fcontext
# else
	.size .jump_fcontext, .-.jump_fcontext
# endif
#endif

#if ABT_CONFIG_THREAD_TYPE == ABT_THREAD_TYPE_DYNAMIC_PROMOTION
.globl init_and_call_fcontext
#if _CALL_ELF == 2
    .text
    .align 2
init_and_call_fcontext:
        addis   %r2, %r12, .TOC.-init_and_call_fcontext@ha
        addi    %r2, %r2, .TOC.-init_and_call_fcontext@l
        .localentry init_and_call_fcontext, . - init_and_call_fcontext
#else
    .section ".opd","aw"
    .align 3
init_and_call_fcontext:
# ifdef _CALL_LINUX
        .quad   .L.init_and_call_fcontext,.TOC.@tocbase,0
        .type   init_and_call_fcontext,@function
        .text
        .align 2
.L.init_and_call_fcontext:
# else
        .hidden .init_and_call_fcontext
        .globl  .init_and_call_fcontext
        .quad   .init_and_call_fcontext,.TOC.@tocbase,0
        .size   init_and_call_fcontext,24
        .type   .init_and_call_fcontext,@function
        .text
        .align 2
.init_and_call_fcontext:
# endif
#endif
    # save current LR, RSP (=R1), and TOC (=R2) in the target stack (=R5)
    # note TOC must be saved when an external function is called.
    # save LR
    mflr  %r0
    std  %r0, -16(%r5)
    # save the current RSP + 0x10
    addi %r10, %r1, 16
    std  %r10, -8(%r5)
    # save TOC
    std  %r2, -24(%r5)

    # reserve space on stack
    subi  %r1, %r1, 528

    std  %r14, 352(%r1)  # save R14
    std  %r15, 360(%r1)  # save R15
    std  %r16, 368(%r1)  # save R16
    std  %r17, 376(%r1)  # save R17
    std  %r18, 384(%r1)  # save R18
    std  %r19, 392(%r1)  # save R19
    std  %r20, 400(%r1)  # save R20
    std  %r21, 408(%r1)  # save R21
    std  %r22, 416(%r1)  # save R22
    std  %r23, 424(%r1)  # save R23
    std  %r24, 432(%r1)  # save R24
    std  %r25, 440(%r1)  # save R25
    std  %r26, 448(%r1)  # save R26
    std  %r27, 456(%r1)  # save R27
    std  %r28, 464(%r1)  # save R28
    std  %r29, 472(%r1)  # save R29
    std  %r30, 480(%r1)  # save R30
    std  %r31, 488(%r1)  # save R31
#if _CALL_ELF != 2
    std  %r2,  496(%r1)  # save TOC
#endif

    # save LR; LR has been loaded to R0 at the beginning of the function
    std  %r0, 512(%r1)
    # save LR as PC
    std  %r0, 520(%r1)
    # save CR
    mfcr  %r0
    std  %r0, 504(%r1)

#if ABTD_FCONTEXT_PRESERVE_FPU
    stfd  %f14, 0(%r1)  # save F14
    stfd  %f15, 8(%r1)  # save F15
    stfd  %f16, 16(%r1)  # save F16
    stfd  %f17, 24(%r1)  # save F17
    stfd  %f18, 32(%r1)  # save F18
    stfd  %f19, 40(%r1)  # save F19
    stfd  %f20, 48(%r1)  # save F20
    stfd  %f21, 56(%r1)  # save F21
    stfd  %f22, 64(%r1)  # save F22
    stfd  %f23, 72(%r1)  # save F23
    stfd  %f24, 80(%r1)  # save F24
    stfd  %f25, 88(%r1)  # save F25
    stfd  %f26, 96(%r1)  # save F26
    stfd  %f27, 104(%r1)  # save F27
    stfd  %f28, 112(%r1)  # save F28
    stfd  %f29, 120(%r1)  # save F29
    stfd  %f30, 128(%r1)  # save F30
    stfd  %f31, 136(%r1)  # save F31
#ifdef __VSX__
    # VSCR can be loaded only to Vn.  To store VSCR as it is a vector, it must
    # be written before saving FPSCR.
    mfvscr %v19           # load VSCR
    li    %r10, 144
    stvx  %v19, %r10, %r1 # save VSCR.  Only the last 32 bits are used
#endif
    mffs  %f0  # load FPSCR
    stfd  %f0, 144(%r1)  # save FPSCR
#ifdef __VSX__
    # OpenPOWER saves V20-V31 (vector units)
    # note stvx cannot take an immediate value as an offset.
    # use general caller-saved registers (r7-r10) to store immediate values
    li    %r7, 160
    stvx  %v20, %r7, %r1   # save V20
    li    %r8, 176
    stvx  %v21, %r8, %r1   # save V21
    li    %r9, 192
    stvx  %v22, %r9, %r1   # save V22
    li    %r10, 208
    stvx  %v23, %r10, %r1  # save V23
    li    %r7, 224
    stvx  %v24, %r7, %r1   # save V24
    li    %r8, 240
    stvx  %v25, %r8, %r1   # save V25
    li    %r9, 256
    stvx  %v26, %r9, %r1   # save V26
    li    %r10, 272
    stvx  %v27, %r10, %r1  # save V27
    li    %r7, 288
    stvx  %v28, %r7, %r1   # save V28
    li    %r8, 304
    stvx  %v29, %r8, %r1   # save V29
    li    %r9, 320
    stvx  %v30, %r9, %r1   # save V30
    li    %r10, 336
    stvx  %v31, %r10, %r1  # save V31
#endif
#endif

    # store RSP (pointing to context-data) in R6 (=fctx)
    std   %r1, 0(%r6)

    # load RSP (pointing to context-data) from R5 (=p_stacktop) and add offset
    # the following is equivalent to
    #   %r1 <= %r5 - 0x30
    #   (%r1) <= %r5
    # R5 must be 16-byte aligned (ABI specification)
    mr    %r1, %r5
    stdu  %r1, -48(%r1)

    # set R4 (=f_thread) to CTR.
    # f_thread can be a global entry point, so R12 must be set as well
    mr    %r12, %r4
    mtctr %r12
    # call CTR (=f_thread); note that arg (=R3) has been already set.
    # note that TOC has been saved at the very beginning of the function
    bctrl

    #
    # - when the thread did not yield, RSP is set to the original one, so ret
    #   jumps to the original control flow.
    #
    #  [0x12345600] : (the original stack frame)
    #   ...
    #  [RSP + 0x28] : the original stack pointer (0x12345600) + 0x10
    #  [RSP + 0x20] : the original LR pointer
    #  [RSP + 0x18] : TOC: this TOC pointer
    #  [RSP + 0x10] : LR: reserved
    #  [RSP + 0x08] : CR: reserved
    #  [RSP + 0x00] : RSP: this stack pointer
    #  [RSP - ~~~~] : used by f_thread
    #
    # - any suspension updates RSP to (p_stacktop - 0x10), and calls the jump
    #   termination function by blr.
    #   RSP is 16-byte aligned (ABI specification).
    #
    #  (p_stacktop = RSP + 0x30)
    #  [RSP + 0x28] : pointing to (p_stacktop - 0x10 (= RSP + 0x20))
    #  [RSP + 0x20] : the address of the termination function.
    #  [RSP + 0x18] : TOC: this TOC pointer
    #  [RSP + 0x10] : LR: reserved
    #  [RSP + 0x08] : CR: reserved
    #  [RSP + 0x00] : RSP: this stack pointer
    #  [RSP - ~~~~] : used by f_thread
    #

    # restore LR from the stack
    # because it might call an external function, set R12 as well
    ld    %r12, 32(%r1)
    mtlr  %r12
    # restore TOC from the stack
    ld    %r2, 24(%r1)
    # restore RSP
    ld    %r1, 40(%r1)
    # RSP is subtracted by 16; RSP points at the original stack pointer if the
    # thread did not yield.  If the thread has been suspended, RSP becomes
    # p_stacktop - 0x20 so that the callee can use 32 bytes above RSP.
    subi  %r1, %r1, 16
    # return
    blr
#if _CALL_ELF == 2
    .size init_and_call_fcontext, .-init_and_call_fcontext
#else
# ifdef _CALL_LINUX
    .size .init_and_call_fcontext, .-.L.init_and_call_fcontext
# else
    .size .init_and_call_fcontext, .-.init_and_call_fcontext
# endif
#endif
#endif

/* Mark that we don't need executable stack.  */
.section .note.GNU-stack,"",%progbits

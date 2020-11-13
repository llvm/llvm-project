/*
            Copyright Oliver Kowalke 2009.
   Distributed under the Boost Software License, Version 1.0.
      (See accompanying file LICENSE_1_0.txt or copy at
            http://www.boost.org/LICENSE_1_0.txt)
*/

/****************************************************************************************
 *                                                                                      *
 *  ----------------------------------------------------------------------------------  *
 *  |    0    |    1    |    2    |    3    |    4     |    5    |    6    |    7    |  *
 *  ----------------------------------------------------------------------------------  *
 *  |   0x0   |   0x4   |   0x8   |   0xc   |   0x10   |   0x14  |   0x18  |   0x1c  |  *
 *  ----------------------------------------------------------------------------------  *
 *  | fc_mxcsr|fc_x87_cw|        R12        |         R13        |        R14        |  *
 *  ----------------------------------------------------------------------------------  *
 *  ----------------------------------------------------------------------------------  *
 *  |    8    |    9    |   10    |   11    |    12    |    13   |    14   |    15   |  *
 *  ----------------------------------------------------------------------------------  *
 *  |   0x20  |   0x24  |   0x28  |  0x2c   |   0x30   |   0x34  |   0x38  |   0x3c  |  *
 *  ----------------------------------------------------------------------------------  *
 *  |        R15        |        RBX        |         RBP        |        RIP        |  *
 *  ----------------------------------------------------------------------------------  *
 *  ----------------------------------------------------------------------------------  *
 *  |    16   |   17    |                                                            |  *
 *  ----------------------------------------------------------------------------------  *
 *  |   0x40  |   0x44  |                                                            |  *
 *  ----------------------------------------------------------------------------------  *
 *  |        EXIT       |                                                            |  *
 *  ----------------------------------------------------------------------------------  *
 *                                                                                      *
 ****************************************************************************************/

#include "abt_config.h"

.text
.globl jump_fcontext
.type jump_fcontext,@function
.align 16
jump_fcontext:
    pushq  %rbp  /* save RBP */
    pushq  %rbx  /* save RBX */
    pushq  %r15  /* save R15 */
    pushq  %r14  /* save R14 */
    pushq  %r13  /* save R13 */
    pushq  %r12  /* save R12 */

    /* prepare stack for FPU */
    leaq  -0x8(%rsp), %rsp

#if ABTD_FCONTEXT_PRESERVE_FPU
    /* save MMX control- and status-word */
    stmxcsr  (%rsp)
    /* save x87 control-word */
    fnstcw   0x4(%rsp)
#endif

    /* store RSP (pointing to context-data) in RDI */
    movq  %rsp, (%rdi)

    /* restore RSP (pointing to context-data) from RSI */
    movq  %rsi, %rsp

#if ABTD_FCONTEXT_PRESERVE_FPU
    /* restore MMX control- and status-word */
    ldmxcsr  (%rsp)
    /* restore x87 control-word */
    fldcw  0x4(%rsp)
#endif

    /* prepare stack for FPU */
    leaq  0x8(%rsp), %rsp

    popq  %r12  /* restrore R12 */
    popq  %r13  /* restrore R13 */
    popq  %r14  /* restrore R14 */
    popq  %r15  /* restrore R15 */
    popq  %rbx  /* restrore RBX */
    popq  %rbp  /* restrore RBP */

    /* restore return-address */
    popq  %r8

    /* use third arg as return-value after jump */
    /* movq  %rdx, %rax */
    /* use third arg as first arg in context function */
    movq  %rdx, %rdi

    /* indirect jump to context */
    jmp  *%r8
.size jump_fcontext,.-jump_fcontext

#if ABT_CONFIG_THREAD_TYPE == ABT_THREAD_TYPE_DYNAMIC_PROMOTION
.text
.globl init_and_call_fcontext
.type init_and_call_fcontext,@function
.align 16
init_and_call_fcontext:
    /* save the current rsp to the new stack,
     * which will be restored by movq 0x8(%rsp), %rsp */
    movq %rsp, -0x8(%rdx)
    /* save callee-saved registers */
    pushq %rbp  /* save RBP */
    pushq %rbx  /* save RBX */
    pushq %r15  /* save R15 */
    pushq %r14  /* save R14 */
    pushq %r13  /* save R13 */
    pushq %r12  /* save R12 */
    /* prepare stack for FPU */
    leaq  -0x8(%rsp), %rsp
#if ABTD_FCONTEXT_PRESERVE_FPU
    /* save MMX control- and status-word */
    stmxcsr  (%rsp)
    /* save x87 control-word */
    fnstcw   0x4(%rsp)
#endif
    /* store RSP in RCX (= fctx) */
    movq %rsp, (%rcx)
    leaq -0x10(%rdx), %rsp
    /* call RSI (= f_thread). RDI (= p_arg) has been already set */
    /* RSP is 16-byte aligned (ABI specification) */
    callq *%rsi
    /* restore original RSP */
    movq 0x8(%rsp), %rsp
    /*
     * - When the thread did not yield, RSP is set to the original one, so ret
     *   jumps to the original control flow.
     * - Any suspension updates RSP to (p_stack - 0x10), so that ret
     *   calls (p_stack - 0x8), which is set to a termination function.
     *   RSP is 16-byte aligned (ABI specification).
     */
    ret
.size init_and_call_fcontext,.-init_and_call_fcontext
#endif

/* Mark that we don't need executable stack.  */
#ifndef __SUNPRO_C
.section .note.GNU-stack,"",%progbits
#else
/* Solaris assembler does not recognize it.  Let's ignore so far. */
#endif

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
 *  | fc_mxcsr|fc_x87_cw|   EDI   |   ESI   |   EBX    |   EBP   |   EIP   |   EXIT  |  *
 *  ----------------------------------------------------------------------------------  *
 *                                                                                      *
 ****************************************************************************************/

#include "abt_config.h"

.text
.globl jump_fcontext
.align 2
.type jump_fcontext,@function
jump_fcontext:
    pushl  %ebp  /* save EBP */
    pushl  %ebx  /* save EBX */
    pushl  %esi  /* save ESI */
    pushl  %edi  /* save EDI */

    /* prepare stack for FPU */
    leal  -0x8(%esp), %esp

#if ABTD_FCONTEXT_PRESERVE_FPU
    /* save MMX control- and status-word */
    stmxcsr  (%esp)
    /* save x87 control-word */
    fnstcw  0x4(%esp)
#endif

    /* first arg of jump_fcontext() == context jumping from */
    movl  0x1c(%esp), %eax

    /* store ESP (pointing to context-data) in EAX */
    movl  %esp, (%eax)

    /* second arg of jump_fcontext() == context jumping to */
    movl  0x20(%esp), %edx

    /* third arg of jump_fcontext() == value to be returned after jump */
    movl  0x24(%esp), %eax

    /* restore ESP (pointing to context-data) from EDX */
    movl  %edx, %esp

#if ABTD_FCONTEXT_PRESERVE_FPU
    /* restore MMX control- and status-word */
    ldmxcsr  (%esp)
    /* restore x87 control-word */
    fldcw  0x4(%esp)
#endif

    /* prepare stack for FPU */
    leal  0x8(%esp), %esp

    popl  %edi  /* restore EDI */
    popl  %esi  /* restore ESI */
    popl  %ebx  /* restore EBX */
    popl  %ebp  /* restore EBP */

    /* restore return-address */
    popl  %edx

    /* use value in EAX (originally third arg) as
     * 1. return value after jump or
     * 2. first arg in context function. */
    movl  %eax, 0x4(%esp)

    /* indirect jump to context */
    jmp  *%edx
.size jump_fcontext,.-jump_fcontext

/* Mark that we don't need executable stack.  */
.section .note.GNU-stack,"",%progbits

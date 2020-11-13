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
.globl _take_fcontext
.align 2
_take_fcontext:
    /* first arg is ignored. */

    /* second arg of take_fcontext() == context jumping to */
    movl  0x8(%esp), %edx

    /* third arg of take_fcontext() == value to be returned after jump */
    movl  0xc(%esp), %eax

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

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

.text
.globl make_fcontext
.align 2
.type make_fcontext,@function
make_fcontext:
    /* first arg of make_fcontext() == top of context-stack */
    movl  0x4(%esp), %eax

    /* reserve space for first argument of context-function
       rax might already point to a 16byte border */
    leal  -0x8(%eax), %eax

    /* shift address in EAX to lower 16 byte boundary */
    andl  $-16, %eax

    /* reserve space for context-data on context-stack */
    /* size for fc_mxcsr .. EIP + return-address for context-function */
    /* on context-function entry: (ESP -0x4) % 8 == 0 */
    leal  -0x20(%eax), %eax

    /* third arg of make_fcontext() == address of context-function */
    movl  0xc(%esp), %edx
    movl  %edx, 0x18(%eax)

    /* save MMX control- and status-word */
    stmxcsr  (%eax)
    /* save x87 control-word */
    fnstcw  0x4(%eax)

    /* compute abs address of label finish */
    call  1f
    /* address of label 1 */
1:  popl  %ecx
    /* compute abs address of label finish */
    addl  $finish-1b, %ecx
    /* save address of finish as return-address for context-function */
    /* will be entered after context-function returns */
    movl  %ecx, 0x1c(%eax)

    ret /* return pointer to context-data */

finish:
    call  2f
    /* address of label 2 */
2:  popl  %ebx
    /* compute address of GOT and store it in EBX */
    addl  $_GLOBAL_OFFSET_TABLE_+[.-2b], %ebx

    /* exit code is zero */
    xorl  %eax, %eax
    movl  %eax, (%esp)
    /* exit application */
    call  _exit@PLT
    hlt
.size make_fcontext,.-make_fcontext

/* Mark that we don't need executable stack.  */
.section .note.GNU-stack,"",%progbits

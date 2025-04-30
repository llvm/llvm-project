/* Function expf vectorized with SSE4.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>
#include "svml_s_expf_data.h"

	.text
ENTRY (_ZGVbN4v_expf_sse4)
/*
   ALGORITHM DESCRIPTION:

     Argument representation:
     M = rint(X*2^k/ln2) = 2^k*N+j
     X = M*ln2/2^k + r = N*ln2 + ln2*(j/2^k) + r
     then -ln2/2^(k+1) < r < ln2/2^(k+1)
     Alternatively:
     M = trunc(X*2^k/ln2)
     then 0 < r < ln2/2^k

     Result calculation:
     exp(X) = exp(N*ln2 + ln2*(j/2^k) + r)
     = 2^N * 2^(j/2^k) * exp(r)
     2^N is calculated by bit manipulation
     2^(j/2^k) is computed from table lookup
     exp(r) is approximated by polynomial

     The table lookup is skipped if k = 0.
     For low accuracy approximation, exp(r) ~ 1 or 1+r.  */

        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $320, %rsp
        movaps    %xmm0, %xmm5
        movq      __svml_sexp_data@GOTPCREL(%rip), %rax
        movups __sInvLn2(%rax), %xmm0

/* m = x*2^k/ln2 + shifter */
        mulps     %xmm5, %xmm0
        movups __sShifter(%rax), %xmm6
        movups __sLn2hi(%rax), %xmm4
        addps     %xmm6, %xmm0

/* n = m - shifter = rint(x*2^k/ln2) */
        movaps    %xmm0, %xmm2

/* remove sign of x by "and" operation */
        movdqu __iAbsMask(%rax), %xmm7
        subps     %xmm6, %xmm2

/* r = x-n*ln2_hi/2^k */
        mulps     %xmm2, %xmm4
        pand      %xmm5, %xmm7

/* compare against threshold */
        pcmpgtd __iDomainRange(%rax), %xmm7
        movups __sLn2lo(%rax), %xmm1

/* set mask for overflow/underflow */
        movmskps  %xmm7, %ecx
        movaps    %xmm5, %xmm7
        movups __sPC5(%rax), %xmm3
        subps     %xmm4, %xmm7

/* r = r-n*ln2_lo/2^k = x - n*ln2/2^k */
        mulps     %xmm1, %xmm2

/* compute 2^N with "shift" */
        movdqu __iBias(%rax), %xmm6
        subps     %xmm2, %xmm7

/* c5*r+c4 */
        mulps     %xmm7, %xmm3
        paddd     %xmm6, %xmm0
        pslld     $23, %xmm0
        addps __sPC4(%rax), %xmm3

/* (c5*r+c4)*r+c3 */
        mulps     %xmm7, %xmm3
        addps __sPC3(%rax), %xmm3

/* ((c5*r+c4)*r+c3)*r+c2 */
        mulps     %xmm7, %xmm3
        addps __sPC2(%rax), %xmm3

/* (((c5*r+c4)*r+c3)*r+c2)*r+c1 */
        mulps     %xmm7, %xmm3
        addps __sPC1(%rax), %xmm3

/* exp(r) = ((((c5*r+c4)*r+c3)*r+c2)*r+c1)*r+c0 */
        mulps     %xmm3, %xmm7
        addps __sPC0(%rax), %xmm7

/* 2^N*exp(r) */
        mulps     %xmm7, %xmm0
        testl     %ecx, %ecx
        jne       .LBL_1_3

.LBL_1_2:
        cfi_remember_state
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_1_3:
        cfi_restore_state
        movups    %xmm5, 192(%rsp)
        movups    %xmm0, 256(%rsp)
        je        .LBL_1_2

        xorb      %dl, %dl
        xorl      %eax, %eax
        movups    %xmm8, 112(%rsp)
        movups    %xmm9, 96(%rsp)
        movups    %xmm10, 80(%rsp)
        movups    %xmm11, 64(%rsp)
        movups    %xmm12, 48(%rsp)
        movups    %xmm13, 32(%rsp)
        movups    %xmm14, 16(%rsp)
        movups    %xmm15, (%rsp)
        movq      %rsi, 136(%rsp)
        movq      %rdi, 128(%rsp)
        movq      %r12, 168(%rsp)
        cfi_offset_rel_rsp (12, 168)
        movb      %dl, %r12b
        movq      %r13, 160(%rsp)
        cfi_offset_rel_rsp (13, 160)
        movl      %ecx, %r13d
        movq      %r14, 152(%rsp)
        cfi_offset_rel_rsp (14, 152)
        movl      %eax, %r14d
        movq      %r15, 144(%rsp)
        cfi_offset_rel_rsp (15, 144)
        cfi_remember_state

.LBL_1_6:
        btl       %r14d, %r13d
        jc        .LBL_1_12

.LBL_1_7:
        lea       1(%r14), %esi
        btl       %esi, %r13d
        jc        .LBL_1_10

.LBL_1_8:
        incb      %r12b
        addl      $2, %r14d
        cmpb      $16, %r12b
        jb        .LBL_1_6

        movups    112(%rsp), %xmm8
        movups    96(%rsp), %xmm9
        movups    80(%rsp), %xmm10
        movups    64(%rsp), %xmm11
        movups    48(%rsp), %xmm12
        movups    32(%rsp), %xmm13
        movups    16(%rsp), %xmm14
        movups    (%rsp), %xmm15
        movq      136(%rsp), %rsi
        movq      128(%rsp), %rdi
        movq      168(%rsp), %r12
        cfi_restore (%r12)
        movq      160(%rsp), %r13
        cfi_restore (%r13)
        movq      152(%rsp), %r14
        cfi_restore (%r14)
        movq      144(%rsp), %r15
        cfi_restore (%r15)
        movups    256(%rsp), %xmm0
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        movss     196(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(expf)

        movss     %xmm0, 260(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        movss     192(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(expf)

        movss     %xmm0, 256(%rsp,%r15,8)
        jmp       .LBL_1_7

END (_ZGVbN4v_expf_sse4)

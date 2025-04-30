/* Function exp vectorized with SSE4.
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
#include "svml_d_exp_data.h"

	.text
ENTRY (_ZGVbN2v_exp_sse4)
/*
   ALGORITHM DESCRIPTION:

     Argument representation:
     N = rint(X*2^k/ln2) = 2^k*M+j
     X = N*ln2/2^k + r = M*ln2 + ln2*(j/2^k) + r
     then -ln2/2^(k+1) < r < ln2/2^(k+1)
     Alternatively:
     N = trunc(X*2^k/ln2)
     then 0 < r < ln2/2^k

     Result calculation:
     exp(X) = exp(M*ln2 + ln2*(j/2^k) + r)
     = 2^M * 2^(j/2^k) * exp(r)
     2^M is calculated by bit manipulation
     2^(j/2^k) is stored in table
     exp(r) is approximated by polynomial.

     The table lookup is skipped if k = 0.  */

        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $320, %rsp
        movaps    %xmm0, %xmm3
        movq      __svml_dexp_data@GOTPCREL(%rip), %r8

/* iAbsX = (int)(lX>>32), lX = *(longlong*)&X */
        pshufd    $221, %xmm3, %xmm7
        movups __dbInvLn2(%r8), %xmm0

/* dK = X*dbInvLn2 */
        mulpd     %xmm3, %xmm0
        movq __iAbsMask(%r8), %xmm5
        movq __iDomainRange(%r8), %xmm6

/* iAbsX = iAbsX&iAbsMask */
        pand      %xmm5, %xmm7

/* iRangeMask = (iAbsX>iDomainRange) */
        pcmpgtd   %xmm6, %xmm7

/* Mask = iRangeMask?1:0, set mask for overflow/underflow */
        movmskps  %xmm7, %eax

/* dN = rint(X*2^k/Ln2) */
        xorps     %xmm7, %xmm7
        movups __dbLn2hi(%r8), %xmm5
        movups __dbLn2lo(%r8), %xmm6
        roundpd   $0, %xmm0, %xmm7

/* dR = X - dN*dbLn2hi, dbLn2hi is 52-8-k hi bits of ln2/2^k */
        mulpd     %xmm7, %xmm5

/* dR = dR - dN*dbLn2lo, dbLn2lo is 40..94 bits of lo part of ln2/2^k */
        mulpd     %xmm6, %xmm7
        movups __dbShifter(%r8), %xmm4

/* dM = X*dbInvLn2+dbShifter */
        addpd     %xmm0, %xmm4
        movaps    %xmm3, %xmm0
        subpd     %xmm5, %xmm0
        subpd     %xmm7, %xmm0
        movups __dPC2(%r8), %xmm5

/* exp(r) = b0+r*(b0+r*(b1+r*b2)) */
        mulpd     %xmm0, %xmm5
        addpd __dPC1(%r8), %xmm5
        mulpd     %xmm0, %xmm5
        movups __dPC0(%r8), %xmm6
        addpd     %xmm6, %xmm5
        mulpd     %xmm5, %xmm0
        movdqu __lIndexMask(%r8), %xmm2

/* lIndex = (*(longlong*)&dM)&lIndexMask, lIndex is the lower K bits of lM */
        movdqa    %xmm2, %xmm1

/* lM = (*(longlong*)&dM)&(~lIndexMask) */
        pandn     %xmm4, %xmm2
        pand      %xmm4, %xmm1

/* lM = lM<<(52-K), 2^M */
        psllq     $42, %xmm2

/* table lookup for dT[j] = 2^(j/2^k) */
        movd      %xmm1, %edx
        pextrw    $4, %xmm1, %ecx
        addpd     %xmm0, %xmm6
        shll      $3, %edx
        shll      $3, %ecx
        movq      (%r8,%rdx), %xmm0
        andl      $3, %eax
        movhpd    (%r8,%rcx), %xmm0

/* 2^(j/2^k) * exp(r) */
        mulpd     %xmm6, %xmm0

/* multiply by 2^M through integer add */
        paddq     %xmm2, %xmm0
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
        movups    %xmm3, 192(%rsp)
        movups    %xmm0, 256(%rsp)
        je        .LBL_1_2

        xorb      %cl, %cl
        xorl      %edx, %edx
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
        movb      %cl, %r12b
        movq      %r13, 160(%rsp)
        cfi_offset_rel_rsp (13, 160)
        movl      %eax, %r13d
        movq      %r14, 152(%rsp)
        cfi_offset_rel_rsp (14, 152)
        movl      %edx, %r14d
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
        shlq      $4, %r15
        movsd     200(%rsp,%r15), %xmm0

        call      JUMPTARGET(exp)

        movsd     %xmm0, 264(%rsp,%r15)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        movsd     192(%rsp,%r15), %xmm0

        call      JUMPTARGET(exp)

        movsd     %xmm0, 256(%rsp,%r15)
        jmp       .LBL_1_7

END (_ZGVbN2v_exp_sse4)

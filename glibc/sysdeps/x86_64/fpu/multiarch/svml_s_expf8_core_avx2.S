/* Function expf vectorized with AVX2.
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
ENTRY(_ZGVdN8v_expf_avx2)
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
        subq      $448, %rsp
        movq      __svml_sexp_data@GOTPCREL(%rip), %rax
        vmovaps   %ymm0, %ymm2
        vmovups __sInvLn2(%rax), %ymm7
        vmovups __sShifter(%rax), %ymm4
        vmovups __sLn2hi(%rax), %ymm3
        vmovups __sPC5(%rax), %ymm1

/* m = x*2^k/ln2 + shifter */
        vfmadd213ps %ymm4, %ymm2, %ymm7

/* n = m - shifter = rint(x*2^k/ln2) */
        vsubps    %ymm4, %ymm7, %ymm0
        vpaddd __iBias(%rax), %ymm7, %ymm4

/* remove sign of x by "and" operation */
        vandps __iAbsMask(%rax), %ymm2, %ymm5

/* compare against threshold */
        vpcmpgtd __iDomainRange(%rax), %ymm5, %ymm6

/* r = x-n*ln2_hi/2^k */
        vmovaps   %ymm2, %ymm5
        vfnmadd231ps %ymm0, %ymm3, %ymm5

/* r = r-n*ln2_lo/2^k = x - n*ln2/2^k */
        vfnmadd132ps __sLn2lo(%rax), %ymm5, %ymm0

/* c5*r+c4 */
        vfmadd213ps __sPC4(%rax), %ymm0, %ymm1

/* (c5*r+c4)*r+c3 */
        vfmadd213ps __sPC3(%rax), %ymm0, %ymm1

/* ((c5*r+c4)*r+c3)*r+c2 */
        vfmadd213ps __sPC2(%rax), %ymm0, %ymm1

/* (((c5*r+c4)*r+c3)*r+c2)*r+c1 */
        vfmadd213ps __sPC1(%rax), %ymm0, %ymm1

/* exp(r) = ((((c5*r+c4)*r+c3)*r+c2)*r+c1)*r+c0 */
        vfmadd213ps __sPC0(%rax), %ymm0, %ymm1

/* set mask for overflow/underflow */
        vmovmskps %ymm6, %ecx

/* compute 2^N with "shift" */
        vpslld    $23, %ymm4, %ymm6

/* 2^N*exp(r) */
        vmulps    %ymm1, %ymm6, %ymm0
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
        vmovups   %ymm2, 320(%rsp)
        vmovups   %ymm0, 384(%rsp)
        je        .LBL_1_2

        xorb      %dl, %dl
        xorl      %eax, %eax
        vmovups   %ymm8, 224(%rsp)
        vmovups   %ymm9, 192(%rsp)
        vmovups   %ymm10, 160(%rsp)
        vmovups   %ymm11, 128(%rsp)
        vmovups   %ymm12, 96(%rsp)
        vmovups   %ymm13, 64(%rsp)
        vmovups   %ymm14, 32(%rsp)
        vmovups   %ymm15, (%rsp)
        movq      %rsi, 264(%rsp)
        movq      %rdi, 256(%rsp)
        movq      %r12, 296(%rsp)
        cfi_offset_rel_rsp (12, 296)
        movb      %dl, %r12b
        movq      %r13, 288(%rsp)
        cfi_offset_rel_rsp (13, 288)
        movl      %ecx, %r13d
        movq      %r14, 280(%rsp)
        cfi_offset_rel_rsp (14, 280)
        movl      %eax, %r14d
        movq      %r15, 272(%rsp)
        cfi_offset_rel_rsp (15, 272)
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

        vmovups   224(%rsp), %ymm8
        vmovups   192(%rsp), %ymm9
        vmovups   160(%rsp), %ymm10
        vmovups   128(%rsp), %ymm11
        vmovups   96(%rsp), %ymm12
        vmovups   64(%rsp), %ymm13
        vmovups   32(%rsp), %ymm14
        vmovups   (%rsp), %ymm15
        vmovups   384(%rsp), %ymm0
        movq      264(%rsp), %rsi
        movq      256(%rsp), %rdi
        movq      296(%rsp), %r12
        cfi_restore (%r12)
        movq      288(%rsp), %r13
        cfi_restore (%r13)
        movq      280(%rsp), %r14
        cfi_restore (%r14)
        movq      272(%rsp), %r15
        cfi_restore (%r15)
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        vmovss    324(%rsp,%r15,8), %xmm0
        vzeroupper

        call      JUMPTARGET(expf)

        vmovss    %xmm0, 388(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        vmovss    320(%rsp,%r15,8), %xmm0
        vzeroupper

        call      JUMPTARGET(expf)

        vmovss    %xmm0, 384(%rsp,%r15,8)
        jmp       .LBL_1_7

END(_ZGVdN8v_expf_avx2)

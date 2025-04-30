/* Function exp vectorized with AVX2.
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
ENTRY (_ZGVdN4v_exp_avx2)
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
     exp(r) is approximated by polynomial

     The table lookup is skipped if k = 0.  */

        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $448, %rsp
        movq      __svml_dexp_data@GOTPCREL(%rip), %rax
        vmovdqa   %ymm0, %ymm2
        vmovupd __dbInvLn2(%rax), %ymm3
        vmovupd __dbShifter(%rax), %ymm1
        vmovupd __lIndexMask(%rax), %ymm4

/* dM = X*dbInvLn2+dbShifter, dbInvLn2 = 2^k/Ln2 */
        vfmadd213pd %ymm1, %ymm2, %ymm3

/* iAbsX = (int)(lX>>32), lX = *(longlong*)&X */
        vextracti128 $1, %ymm2, %xmm5
        vshufps   $221, %xmm5, %xmm2, %xmm6

/* iAbsX = iAbsX&iAbsMask */
        vandps __iAbsMask(%rax), %xmm6, %xmm7

/* dN = dM-dbShifter, dN = rint(X*2^k/Ln2) */
        vsubpd    %ymm1, %ymm3, %ymm6

/* iRangeMask = (iAbsX>iDomainRange) */
        vpcmpgtd __iDomainRange(%rax), %xmm7, %xmm0
        vmovupd __dbLn2hi(%rax), %ymm1
        vmovupd __dPC0(%rax), %ymm7

/* Mask = iRangeMask?1:0, set mask for overflow/underflow */
        vmovmskps %xmm0, %ecx
        vmovupd __dPC2(%rax), %ymm0

/* dR = X - dN*dbLn2hi, dbLn2hi is 52-8-k hi bits of ln2/2^k */
        vmovdqa   %ymm2, %ymm5
        vfnmadd231pd %ymm6, %ymm1, %ymm5

/* dR = dR - dN*dbLn2lo, dbLn2lo is 40..94 bits of lo part of ln2/2^k */
        vfnmadd132pd __dbLn2lo(%rax), %ymm5, %ymm6

/* exp(r) = b0+r*(b0+r*(b1+r*b2)) */
        vfmadd213pd __dPC1(%rax), %ymm6, %ymm0
        vfmadd213pd %ymm7, %ymm6, %ymm0
        vfmadd213pd %ymm7, %ymm6, %ymm0

/* lIndex = (*(longlong*)&dM)&lIndexMask, lIndex is the lower K bits of lM */
        vandps    %ymm4, %ymm3, %ymm1

/* table lookup for dT[j] = 2^(j/2^k) */
        vxorpd    %ymm6, %ymm6, %ymm6
        vpcmpeqd  %ymm5, %ymm5, %ymm5
        vgatherqpd %ymm5, (%rax,%ymm1,8), %ymm6

/* lM = (*(longlong*)&dM)&(~lIndexMask) */
        vpandn    %ymm3, %ymm4, %ymm3

/* 2^(j/2^k) * exp(r) */
        vmulpd    %ymm0, %ymm6, %ymm0

/* lM = lM<<(52-K), 2^M */
        vpsllq    $42, %ymm3, %ymm4

/* multiply by 2^M through integer add */
        vpaddq    %ymm4, %ymm0, %ymm0
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
        vmovupd   %ymm2, 320(%rsp)
        vmovupd   %ymm0, 384(%rsp)
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
        vmovupd   384(%rsp), %ymm0
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
        shlq      $4, %r15
        vmovsd    328(%rsp,%r15), %xmm0
        vzeroupper

        call      JUMPTARGET(exp)

        vmovsd    %xmm0, 392(%rsp,%r15)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    320(%rsp,%r15), %xmm0
        vzeroupper

        call      JUMPTARGET(exp)

        vmovsd    %xmm0, 384(%rsp,%r15)
        jmp       .LBL_1_7

END (_ZGVdN4v_exp_avx2)

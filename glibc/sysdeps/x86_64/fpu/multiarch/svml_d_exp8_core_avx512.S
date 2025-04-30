/* Function exp vectorized with AVX-512. KNL and SKX versions.
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
#include "svml_d_wrapper_impl.h"

	.text
ENTRY (_ZGVeN8v_exp_knl)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512 _ZGVdN4v_exp
#else
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
        subq      $1280, %rsp
        movq      __svml_dexp_data@GOTPCREL(%rip), %rax

/* dR = X - dN*dbLn2hi, dbLn2hi is 52-8-k hi bits of ln2/2^k */
        vmovaps   %zmm0, %zmm8

/* iAbsX = (int)(lX>>32), lX = *(longlong*)&X */
        vpsrlq    $32, %zmm0, %zmm1

/* iAbsX = iAbsX&iAbsMask */
        movl      $255, %edx
        vpmovqd   %zmm1, %ymm2
        kmovw     %edx, %k2

/* iRangeMask = (iAbsX>iDomainRange) */
        movl      $-1, %ecx

/* table lookup for dT[j] = 2^(j/2^k) */
        vpxord    %zmm11, %zmm11, %zmm11
        vmovups __dbInvLn2(%rax), %zmm5
        vmovups __dbLn2hi(%rax), %zmm7
        kxnorw    %k3, %k3, %k3

/* dM = X*dbInvLn2+dbShifter, dbInvLn2 = 2^k/Ln2 */
        vfmadd213pd __dbShifter(%rax), %zmm0, %zmm5
        vmovups __dPC2(%rax), %zmm12

/* dN = dM-dbShifter, dN = rint(X*2^k/Ln2) */
        vsubpd __dbShifter(%rax), %zmm5, %zmm9
        vmovups __lIndexMask(%rax), %zmm4
        vfnmadd231pd %zmm9, %zmm7, %zmm8
        vpandd __iAbsMask(%rax), %zmm2, %zmm2{%k2}

/* lIndex = (*(longlong*)&dM)&lIndexMask, lIndex is the lower K bits of lM */
        vpandq    %zmm4, %zmm5, %zmm10
        vgatherqpd (%rax,%zmm10,8), %zmm11{%k3}
        vpcmpgtd __iDomainRange(%rax), %zmm2, %k1{%k2}

/* lM = (*(longlong*)&dM)&(~lIndexMask) */
        vpandnq   %zmm5, %zmm4, %zmm6
        vpbroadcastd %ecx, %zmm3{%k1}{z}

/* lM = lM<<(52-K), 2^M */
        vpsllq    $42, %zmm6, %zmm14

/* dR = dR - dN*dbLn2lo, dbLn2lo is 40..94 bits of lo part of ln2/2^k */
        vfnmadd132pd __dbLn2lo(%rax), %zmm8, %zmm9

/* Mask = iRangeMask?1:0, set mask for overflow/underflow */
        vptestmd  %zmm3, %zmm3, %k0{%k2}

/* exp(r) = b0+r*(b0+r*(b1+r*b2)) */
        vfmadd213pd __dPC1(%rax), %zmm9, %zmm12
        kmovw     %k0, %ecx
        movzbl    %cl, %ecx
        vfmadd213pd __dPC0(%rax), %zmm9, %zmm12
        vfmadd213pd __dPC0(%rax), %zmm9, %zmm12

/* 2^(j/2^k) * exp(r) */
        vmulpd    %zmm12, %zmm11, %zmm13

/* multiply by 2^M through integer add */
        vpaddq    %zmm14, %zmm13, %zmm1
        testl     %ecx, %ecx
        jne       .LBL_1_3

.LBL_1_2:
        cfi_remember_state
        vmovaps   %zmm1, %zmm0
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_1_3:
        cfi_restore_state
        vmovups   %zmm0, 1152(%rsp)
        vmovups   %zmm1, 1216(%rsp)
        je        .LBL_1_2

        xorb      %dl, %dl
        kmovw     %k4, 1048(%rsp)
        xorl      %eax, %eax
        kmovw     %k5, 1040(%rsp)
        kmovw     %k6, 1032(%rsp)
        kmovw     %k7, 1024(%rsp)
        vmovups   %zmm16, 960(%rsp)
        vmovups   %zmm17, 896(%rsp)
        vmovups   %zmm18, 832(%rsp)
        vmovups   %zmm19, 768(%rsp)
        vmovups   %zmm20, 704(%rsp)
        vmovups   %zmm21, 640(%rsp)
        vmovups   %zmm22, 576(%rsp)
        vmovups   %zmm23, 512(%rsp)
        vmovups   %zmm24, 448(%rsp)
        vmovups   %zmm25, 384(%rsp)
        vmovups   %zmm26, 320(%rsp)
        vmovups   %zmm27, 256(%rsp)
        vmovups   %zmm28, 192(%rsp)
        vmovups   %zmm29, 128(%rsp)
        vmovups   %zmm30, 64(%rsp)
        vmovups   %zmm31, (%rsp)
        movq      %rsi, 1064(%rsp)
        movq      %rdi, 1056(%rsp)
        movq      %r12, 1096(%rsp)
        cfi_offset_rel_rsp (12, 1096)
        movb      %dl, %r12b
        movq      %r13, 1088(%rsp)
        cfi_offset_rel_rsp (13, 1088)
        movl      %ecx, %r13d
        movq      %r14, 1080(%rsp)
        cfi_offset_rel_rsp (14, 1080)
        movl      %eax, %r14d
        movq      %r15, 1072(%rsp)
        cfi_offset_rel_rsp (15, 1072)
        cfi_remember_state

.LBL_1_6:
        btl       %r14d, %r13d
        jc        .LBL_1_12

.LBL_1_7:
        lea       1(%r14), %esi
        btl       %esi, %r13d
        jc        .LBL_1_10

.LBL_1_8:
        addb      $1, %r12b
        addl      $2, %r14d
        cmpb      $16, %r12b
        jb        .LBL_1_6

        kmovw     1048(%rsp), %k4
        movq      1064(%rsp), %rsi
        kmovw     1040(%rsp), %k5
        movq      1056(%rsp), %rdi
        kmovw     1032(%rsp), %k6
        movq      1096(%rsp), %r12
        cfi_restore (%r12)
        movq      1088(%rsp), %r13
        cfi_restore (%r13)
        kmovw     1024(%rsp), %k7
        vmovups   960(%rsp), %zmm16
        vmovups   896(%rsp), %zmm17
        vmovups   832(%rsp), %zmm18
        vmovups   768(%rsp), %zmm19
        vmovups   704(%rsp), %zmm20
        vmovups   640(%rsp), %zmm21
        vmovups   576(%rsp), %zmm22
        vmovups   512(%rsp), %zmm23
        vmovups   448(%rsp), %zmm24
        vmovups   384(%rsp), %zmm25
        vmovups   320(%rsp), %zmm26
        vmovups   256(%rsp), %zmm27
        vmovups   192(%rsp), %zmm28
        vmovups   128(%rsp), %zmm29
        vmovups   64(%rsp), %zmm30
        vmovups   (%rsp), %zmm31
        movq      1080(%rsp), %r14
        cfi_restore (%r14)
        movq      1072(%rsp), %r15
        cfi_restore (%r15)
        vmovups   1216(%rsp), %zmm1
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    1160(%rsp,%r15), %xmm0
        call      JUMPTARGET(exp)
        vmovsd    %xmm0, 1224(%rsp,%r15)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    1152(%rsp,%r15), %xmm0
        call      JUMPTARGET(exp)
        vmovsd    %xmm0, 1216(%rsp,%r15)
        jmp       .LBL_1_7
#endif
END (_ZGVeN8v_exp_knl)

ENTRY (_ZGVeN8v_exp_skx)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512 _ZGVdN4v_exp
#else
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
        subq      $1280, %rsp
        movq      __svml_dexp_data@GOTPCREL(%rip), %rax

/* table lookup for dT[j] = 2^(j/2^k) */
        kxnorw    %k1, %k1, %k1

/* iAbsX = (int)(lX>>32), lX = *(longlong*)&X */
        vpsrlq    $32, %zmm0, %zmm1
        vmovups __dbInvLn2(%rax), %zmm7
        vmovups __dbShifter(%rax), %zmm5
        vmovups __lIndexMask(%rax), %zmm6
        vmovups __dbLn2hi(%rax), %zmm9
        vmovups __dPC0(%rax), %zmm12

/* dM = X*dbInvLn2+dbShifter, dbInvLn2 = 2^k/Ln2 */
        vfmadd213pd %zmm5, %zmm0, %zmm7
        vpmovqd   %zmm1, %ymm2

/* dN = dM-dbShifter, dN = rint(X*2^k/Ln2) */
        vsubpd    %zmm5, %zmm7, %zmm11

/* iAbsX = iAbsX&iAbsMask */
        vpand __iAbsMask(%rax), %ymm2, %ymm3

/* dR = X - dN*dbLn2hi, dbLn2hi is 52-8-k hi bits of ln2/2^k */
        vmovaps   %zmm0, %zmm10
        vfnmadd231pd %zmm11, %zmm9, %zmm10
        vmovups __dPC2(%rax), %zmm9

/* dR = dR - dN*dbLn2lo, dbLn2lo is 40..94 bits of lo part of ln2/2^k */
        vfnmadd132pd __dbLn2lo(%rax), %zmm10, %zmm11

/* exp(r) = b0+r*(b0+r*(b1+r*b2)) */
        vfmadd213pd __dPC1(%rax), %zmm11, %zmm9
        vfmadd213pd %zmm12, %zmm11, %zmm9
        vfmadd213pd %zmm12, %zmm11, %zmm9

/* iRangeMask = (iAbsX>iDomainRange) */
        vpcmpgtd __iDomainRange(%rax), %ymm3, %ymm4

/* Mask = iRangeMask?1:0, set mask for overflow/underflow */
        vmovmskps %ymm4, %ecx

/* lIndex = (*(longlong*)&dM)&lIndexMask, lIndex is the lower K bits of lM */
        vpandq    %zmm6, %zmm7, %zmm13
        vpmovqd   %zmm13, %ymm14
        vpxord    %zmm15, %zmm15, %zmm15
        vgatherdpd (%rax,%ymm14,8), %zmm15{%k1}

/* 2^(j/2^k) * exp(r) */
        vmulpd    %zmm9, %zmm15, %zmm10

/* lM = (*(longlong*)&dM)&(~lIndexMask) */
        vpandnq   %zmm7, %zmm6, %zmm8

/* lM = lM<<(52-K), 2^M */
        vpsllq    $42, %zmm8, %zmm1

/* multiply by 2^M through integer add */
        vpaddq    %zmm1, %zmm10, %zmm1
        testl     %ecx, %ecx
        jne       .LBL_2_3

.LBL_2_2:
        cfi_remember_state
        vmovaps   %zmm1, %zmm0
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_2_3:
        cfi_restore_state
        vmovups   %zmm0, 1152(%rsp)
        vmovups   %zmm1, 1216(%rsp)
        je        .LBL_2_2

        xorb      %dl, %dl
        xorl      %eax, %eax
        kmovw     %k4, 1048(%rsp)
        kmovw     %k5, 1040(%rsp)
        kmovw     %k6, 1032(%rsp)
        kmovw     %k7, 1024(%rsp)
        vmovups   %zmm16, 960(%rsp)
        vmovups   %zmm17, 896(%rsp)
        vmovups   %zmm18, 832(%rsp)
        vmovups   %zmm19, 768(%rsp)
        vmovups   %zmm20, 704(%rsp)
        vmovups   %zmm21, 640(%rsp)
        vmovups   %zmm22, 576(%rsp)
        vmovups   %zmm23, 512(%rsp)
        vmovups   %zmm24, 448(%rsp)
        vmovups   %zmm25, 384(%rsp)
        vmovups   %zmm26, 320(%rsp)
        vmovups   %zmm27, 256(%rsp)
        vmovups   %zmm28, 192(%rsp)
        vmovups   %zmm29, 128(%rsp)
        vmovups   %zmm30, 64(%rsp)
        vmovups   %zmm31, (%rsp)
        movq      %rsi, 1064(%rsp)
        movq      %rdi, 1056(%rsp)
        movq      %r12, 1096(%rsp)
        cfi_offset_rel_rsp (12, 1096)
        movb      %dl, %r12b
        movq      %r13, 1088(%rsp)
        cfi_offset_rel_rsp (13, 1088)
        movl      %ecx, %r13d
        movq      %r14, 1080(%rsp)
        cfi_offset_rel_rsp (14, 1080)
        movl      %eax, %r14d
        movq      %r15, 1072(%rsp)
        cfi_offset_rel_rsp (15, 1072)
        cfi_remember_state

.LBL_2_6:
        btl       %r14d, %r13d
        jc        .LBL_2_12

.LBL_2_7:
        lea       1(%r14), %esi
        btl       %esi, %r13d
        jc        .LBL_2_10

.LBL_2_8:
        incb      %r12b
        addl      $2, %r14d
        cmpb      $16, %r12b
        jb        .LBL_2_6

        kmovw     1048(%rsp), %k4
        kmovw     1040(%rsp), %k5
        kmovw     1032(%rsp), %k6
        kmovw     1024(%rsp), %k7
        vmovups   960(%rsp), %zmm16
        vmovups   896(%rsp), %zmm17
        vmovups   832(%rsp), %zmm18
        vmovups   768(%rsp), %zmm19
        vmovups   704(%rsp), %zmm20
        vmovups   640(%rsp), %zmm21
        vmovups   576(%rsp), %zmm22
        vmovups   512(%rsp), %zmm23
        vmovups   448(%rsp), %zmm24
        vmovups   384(%rsp), %zmm25
        vmovups   320(%rsp), %zmm26
        vmovups   256(%rsp), %zmm27
        vmovups   192(%rsp), %zmm28
        vmovups   128(%rsp), %zmm29
        vmovups   64(%rsp), %zmm30
        vmovups   (%rsp), %zmm31
        vmovups   1216(%rsp), %zmm1
        movq      1064(%rsp), %rsi
        movq      1056(%rsp), %rdi
        movq      1096(%rsp), %r12
        cfi_restore (%r12)
        movq      1088(%rsp), %r13
        cfi_restore (%r13)
        movq      1080(%rsp), %r14
        cfi_restore (%r14)
        movq      1072(%rsp), %r15
        cfi_restore (%r15)
        jmp       .LBL_2_2

.LBL_2_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    1160(%rsp,%r15), %xmm0
        vzeroupper
        vmovsd    1160(%rsp,%r15), %xmm0
        call      JUMPTARGET(exp)
        vmovsd    %xmm0, 1224(%rsp,%r15)
        jmp       .LBL_2_8

.LBL_2_12:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    1152(%rsp,%r15), %xmm0
        vzeroupper
        vmovsd    1152(%rsp,%r15), %xmm0
        call      JUMPTARGET(exp)
        vmovsd    %xmm0, 1216(%rsp,%r15)
        jmp       .LBL_2_7

#endif
END (_ZGVeN8v_exp_skx)

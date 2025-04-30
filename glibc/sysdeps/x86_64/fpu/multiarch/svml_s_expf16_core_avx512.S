/* Function expf vectorized with AVX-512. KNL and SKX versions.
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
#include "svml_s_wrapper_impl.h"

	.text
ENTRY (_ZGVeN16v_expf_knl)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512 _ZGVdN8v_expf
#else
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
        subq      $1280, %rsp
        movq      __svml_sexp_data@GOTPCREL(%rip), %rax

/* r = x-n*ln2_hi/2^k */
        vmovaps   %zmm0, %zmm6

/* compare against threshold */
        movl      $-1, %ecx
        vmovups   __sInvLn2(%rax), %zmm3
        vmovups   __sLn2hi(%rax), %zmm5

/* m = x*2^k/ln2 + shifter */
        vfmadd213ps __sShifter(%rax), %zmm0, %zmm3
        vmovups     __sPC5(%rax), %zmm9

/* n = m - shifter = rint(x*2^k/ln2) */
        vsubps    __sShifter(%rax), %zmm3, %zmm7

/* remove sign of x by "and" operation */
        vpandd   __iAbsMask(%rax), %zmm0, %zmm1
        vpaddd   __iBias(%rax), %zmm3, %zmm4
        vpcmpgtd __iDomainRange(%rax), %zmm1, %k1

/* compute 2^N with "shift" */
        vpslld       $23, %zmm4, %zmm8
        vfnmadd231ps %zmm7, %zmm5, %zmm6
        vpbroadcastd %ecx, %zmm2{%k1}{z}

/* r = r-n*ln2_lo/2^k = x - n*ln2/2^k */
        vfnmadd132ps __sLn2lo(%rax), %zmm6, %zmm7

/* set mask for overflow/underflow */
        vptestmd  %zmm2, %zmm2, %k0
        kmovw     %k0, %ecx

/* c5*r+c4 */
        vfmadd213ps __sPC4(%rax), %zmm7, %zmm9

/* (c5*r+c4)*r+c3 */
        vfmadd213ps __sPC3(%rax), %zmm7, %zmm9

/* ((c5*r+c4)*r+c3)*r+c2 */
        vfmadd213ps __sPC2(%rax), %zmm7, %zmm9

/* (((c5*r+c4)*r+c3)*r+c2)*r+c1 */
        vfmadd213ps __sPC1(%rax), %zmm7, %zmm9

/* exp(r) = ((((c5*r+c4)*r+c3)*r+c2)*r+c1)*r+c0 */
        vfmadd213ps __sPC0(%rax), %zmm7, %zmm9

/* 2^N*exp(r) */
        vmulps    %zmm9, %zmm8, %zmm1
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
        vmovss    1156(%rsp,%r15,8), %xmm0
        call      JUMPTARGET(expf)
        vmovss    %xmm0, 1220(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        vmovss    1152(%rsp,%r15,8), %xmm0
        call      JUMPTARGET(expf)
        vmovss    %xmm0, 1216(%rsp,%r15,8)
        jmp       .LBL_1_7

#endif
END (_ZGVeN16v_expf_knl)

ENTRY (_ZGVeN16v_expf_skx)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512 _ZGVdN8v_expf
#else
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
        subq      $1280, %rsp
        movq      __svml_sexp_data@GOTPCREL(%rip), %rax

/* r = x-n*ln2_hi/2^k */
        vmovaps   %zmm0, %zmm7

/* compare against threshold */
        vmovups   .L_2il0floatpacket.13(%rip), %zmm3
        vmovups __sInvLn2(%rax), %zmm4
        vmovups __sShifter(%rax), %zmm1
        vmovups __sLn2hi(%rax), %zmm6
        vmovups __sPC5(%rax), %zmm10

/* m = x*2^k/ln2 + shifter */
        vfmadd213ps %zmm1, %zmm0, %zmm4

/* n = m - shifter = rint(x*2^k/ln2) */
        vsubps    %zmm1, %zmm4, %zmm8
        vpaddd __iBias(%rax), %zmm4, %zmm5
        vfnmadd231ps %zmm8, %zmm6, %zmm7

/* compute 2^N with "shift" */
        vpslld    $23, %zmm5, %zmm9

/* r = r-n*ln2_lo/2^k = x - n*ln2/2^k */
        vfnmadd132ps __sLn2lo(%rax), %zmm7, %zmm8

/* c5*r+c4 */
        vfmadd213ps __sPC4(%rax), %zmm8, %zmm10

/* (c5*r+c4)*r+c3 */
        vfmadd213ps __sPC3(%rax), %zmm8, %zmm10

/* ((c5*r+c4)*r+c3)*r+c2 */
        vfmadd213ps __sPC2(%rax), %zmm8, %zmm10

/* (((c5*r+c4)*r+c3)*r+c2)*r+c1 */
        vfmadd213ps __sPC1(%rax), %zmm8, %zmm10

/* exp(r) = ((((c5*r+c4)*r+c3)*r+c2)*r+c1)*r+c0 */
        vfmadd213ps __sPC0(%rax), %zmm8, %zmm10

/* 2^N*exp(r) */
        vmulps    %zmm10, %zmm9, %zmm1

/* remove sign of x by "and" operation */
        vpandd __iAbsMask(%rax), %zmm0, %zmm2
        vpcmpd    $2, __iDomainRange(%rax), %zmm2, %k1
        vpandnd   %zmm2, %zmm2, %zmm3{%k1}

/* set mask for overflow/underflow */
        vptestmd  %zmm3, %zmm3, %k0
        kmovw     %k0, %ecx
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
        vmovss    1156(%rsp,%r15,8), %xmm0
        vzeroupper
        vmovss    1156(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(expf)

        vmovss    %xmm0, 1220(%rsp,%r15,8)
        jmp       .LBL_2_8

.LBL_2_12:
        movzbl    %r12b, %r15d
        vmovss    1152(%rsp,%r15,8), %xmm0
        vzeroupper
        vmovss    1152(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(expf)

        vmovss    %xmm0, 1216(%rsp,%r15,8)
        jmp       .LBL_2_7

#endif
END (_ZGVeN16v_expf_skx)

	.section .rodata, "a"
.L_2il0floatpacket.13:
	.long	0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff
	.type	.L_2il0floatpacket.13,@object

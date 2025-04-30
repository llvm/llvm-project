/* Function logf vectorized with AVX2.
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
#include "svml_s_logf_data.h"

	.text
ENTRY(_ZGVdN8v_logf_avx2)
/*
   ALGORITHM DESCRIPTION:

    log(x) = exponent_x*log(2) + log(mantissa_x),         if mantissa_x<4/3
    log(x) = (exponent_x+1)*log(2) + log(0.5*mantissa_x), if mantissa_x>4/3

    R = mantissa_x - 1,     if mantissa_x<4/3
    R = 0.5*mantissa_x - 1, if mantissa_x>4/3
    |R|< 1/3

    log(1+R) is approximated as a polynomial: degree 9 for 1-ulp,
    degree 7 for 4-ulp, degree 3 for half-precision.  */

        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $448, %rsp
        movq      __svml_slog_data@GOTPCREL(%rip), %rax
        vmovaps   %ymm0, %ymm2
        vmovups _iBrkValue(%rax), %ymm6
        vmovups _iLoRange(%rax), %ymm1
/* check for working range,
   set special argument mask (denormals/zero/Inf/NaN) */
        vpaddd _iHiDelta(%rax), %ymm2, %ymm7

/* reduction: compute r,n */
        vpsubd    %ymm6, %ymm2, %ymm4

/* exponent_x (mantissa_x<4/3) or exponent_x+1 (mantissa_x>4/3) */
        vpsrad    $23, %ymm4, %ymm3
        vpand _iOffExpoMask(%rax), %ymm4, %ymm5
        vmovups _sPoly_7(%rax), %ymm4
        vcvtdq2ps %ymm3, %ymm0

/* mantissa_x (mantissa_x<4/3), or 0.5*mantissa_x (mantissa_x>4/3) */
        vpaddd    %ymm6, %ymm5, %ymm3

/* reduced argument R */
        vsubps _sOne(%rax), %ymm3, %ymm5

/* polynomial evaluation starts here */
        vfmadd213ps _sPoly_6(%rax), %ymm5, %ymm4
        vfmadd213ps _sPoly_5(%rax), %ymm5, %ymm4
        vfmadd213ps _sPoly_4(%rax), %ymm5, %ymm4
        vfmadd213ps _sPoly_3(%rax), %ymm5, %ymm4
        vfmadd213ps _sPoly_2(%rax), %ymm5, %ymm4
        vfmadd213ps _sPoly_1(%rax), %ymm5, %ymm4
        vmulps    %ymm5, %ymm4, %ymm6

/* polynomial evaluation end */
        vfmadd213ps %ymm5, %ymm5, %ymm6
        vpcmpgtd  %ymm7, %ymm1, %ymm1
        vmovmskps %ymm1, %ecx

/* final reconstruction:
   add exponent_value*log2 to polynomial result */
        vfmadd132ps _sLn2(%rax), %ymm6, %ymm0
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

        call      JUMPTARGET(logf)

        vmovss    %xmm0, 388(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        vmovss    320(%rsp,%r15,8), %xmm0
        vzeroupper

        call      JUMPTARGET(logf)

        vmovss    %xmm0, 384(%rsp,%r15,8)
        jmp       .LBL_1_7

END(_ZGVdN8v_logf_avx2)

/* Function log vectorized with AVX2.
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
#include "svml_d_log_data.h"

	.text
ENTRY (_ZGVdN4v_log_avx2)
/* ALGORITHM DESCRIPTION:

    log(x) = -log(Rcp) + log(Rcp*x),
    where Rcp ~ 1/x (accuracy ~9 bits, obtained by rounding
    HW approximation to 1+9 mantissa bits)

    Reduced argument R=Rcp*x-1 is used to approximate log(1+R) as polynomial

    log(Rcp) = exponent_Rcp*log(2) + log(mantissa_Rcp)
      -log(mantissa_Rcp) is obtained from a lookup table,
      accessed by a 9-bit index
 */
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $448, %rsp
        movq      __svml_dlog_data@GOTPCREL(%rip), %rax
        vmovdqa   %ymm0, %ymm5

/* isolate exponent bits */
        vpsrlq    $20, %ymm5, %ymm0

/* preserve mantissa, set input exponent to 2^(-10) */
        vandpd _ExpMask(%rax), %ymm5, %ymm6
        vorpd _Two10(%rax), %ymm6, %ymm4

/* reciprocal approximation good to at least 11 bits */
        vcvtpd2ps %ymm4, %xmm7
        vrcpps    %xmm7, %xmm1
        vcmplt_oqpd _MinNorm(%rax), %ymm5, %ymm7
        vcvtps2pd %xmm1, %ymm3
        vcmpnle_uqpd _MaxNorm(%rax), %ymm5, %ymm1
        vextracti128 $1, %ymm0, %xmm2
        vshufps   $221, %xmm2, %xmm0, %xmm6

/* round reciprocal to nearest integer, will have 1+9 mantissa bits */
        vroundpd  $0, %ymm3, %ymm2

/* convert biased exponent to DP format */
        vcvtdq2pd %xmm6, %ymm0

/* combine and get argument value range mask */
        vorpd     %ymm1, %ymm7, %ymm3
        vmovupd _One(%rax), %ymm1
        vmovmskpd %ymm3, %ecx

/* calculate index for table lookup */
        vpsrlq    $40, %ymm2, %ymm3

/* argument reduction started:  R = Mantissa*Rcp - 1 */
        vfmsub213pd %ymm1, %ymm2, %ymm4
        vcmpgt_oqpd _Threshold(%rax), %ymm2, %ymm2
        vpcmpeqd  %ymm6, %ymm6, %ymm6
        vxorpd    %ymm1, %ymm1, %ymm1
        vgatherqpd %ymm6, _LogRcp_lookup(%rax,%ymm3), %ymm1

/* exponent*log(2.0) */
        vmovupd _poly_coeff_1(%rax), %ymm6
        vmulpd    %ymm4, %ymm4, %ymm3

/* polynomial computation */
        vfmadd213pd _poly_coeff_2(%rax), %ymm4, %ymm6
        vandpd _Bias(%rax), %ymm2, %ymm7
        vorpd _Bias1(%rax), %ymm7, %ymm2

/*
   Table stores -log(0.5*mantissa) for larger mantissas,
   adjust exponent accordingly
 */
        vsubpd    %ymm2, %ymm0, %ymm0
        vmovupd _poly_coeff_3(%rax), %ymm2
        vfmadd213pd _poly_coeff_4(%rax), %ymm4, %ymm2
        vfmadd213pd %ymm2, %ymm3, %ymm6

/*
   reconstruction:
   (exponent*log(2)) + (LogRcp + (R+poly))
 */
        vfmadd213pd %ymm4, %ymm3, %ymm6
        vaddpd    %ymm1, %ymm6, %ymm4
        vfmadd132pd _L2(%rax), %ymm4, %ymm0
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
        vmovupd   %ymm5, 320(%rsp)
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

        call      JUMPTARGET(log)

        vmovsd    %xmm0, 392(%rsp,%r15)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    320(%rsp,%r15), %xmm0
        vzeroupper

        call      JUMPTARGET(log)

        vmovsd    %xmm0, 384(%rsp,%r15)
        jmp       .LBL_1_7

END (_ZGVdN4v_log_avx2)

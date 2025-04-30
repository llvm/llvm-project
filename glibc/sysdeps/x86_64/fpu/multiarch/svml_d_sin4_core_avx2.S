/* Function sin vectorized with AVX2.
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
#include "svml_d_trig_data.h"

	.text
ENTRY (_ZGVdN4v_sin_avx2)
/* ALGORITHM DESCRIPTION:

      ( low accuracy ( < 4ulp ) or enhanced performance
      ( half of correct mantissa ) implementation )

     Argument representation:
     arg = N*Pi + R

     Result calculation:
     sin(arg) = sin(N*Pi + R) = (-1)^N * sin(R)
     sin(R) is approximated by corresponding polynomial
 */
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $448, %rsp
        movq      __svml_d_trig_data@GOTPCREL(%rip), %rax
        vmovdqa   %ymm0, %ymm4
        vmovupd __dAbsMask(%rax), %ymm2
        vmovupd __dInvPI(%rax), %ymm6
        vmovupd __dRShifter(%rax), %ymm5
        vmovupd __dPI1_FMA(%rax), %ymm7
/*
  ARGUMENT RANGE REDUCTION:
  X' = |X|
 */
        vandpd    %ymm2, %ymm4, %ymm3

/* Y = X'*InvPi + RS : right shifter add */
        vfmadd213pd %ymm5, %ymm3, %ymm6

/* N = Y - RS : right shifter sub */
        vsubpd    %ymm5, %ymm6, %ymm1

/* SignRes = Y<<63 : shift LSB to MSB place for result sign */
        vpsllq    $63, %ymm6, %ymm5

/* R = X' - N*Pi1 */
        vmovapd   %ymm3, %ymm0
        vfnmadd231pd %ymm1, %ymm7, %ymm0
        vcmpnle_uqpd __dRangeVal(%rax), %ymm3, %ymm3

/* R = R - N*Pi2 */
        vfnmadd231pd __dPI2_FMA(%rax), %ymm1, %ymm0

/* R = R - N*Pi3 */
        vfnmadd132pd __dPI3_FMA(%rax), %ymm0, %ymm1

/*
  POLYNOMIAL APPROXIMATION:
  R2 = R*R
 */
        vmulpd    %ymm1, %ymm1, %ymm0

/* R = R^SignRes : update sign of reduced argument */
        vxorpd    %ymm5, %ymm1, %ymm6
        vmovupd   __dC7_sin(%rax), %ymm1
        vfmadd213pd __dC6_sin(%rax), %ymm0, %ymm1
        vfmadd213pd __dC5_sin(%rax), %ymm0, %ymm1
        vfmadd213pd __dC4_sin(%rax), %ymm0, %ymm1

/* Poly = C3+R2*(C4+R2*(C5+R2*(C6+R2*C7))) */
        vfmadd213pd __dC3_sin(%rax), %ymm0, %ymm1

/* Poly = R2*(C1+R2*(C2+R2*Poly)) */
        vfmadd213pd __dC2_sin(%rax), %ymm0, %ymm1
        vfmadd213pd __dC1_sin(%rax), %ymm0, %ymm1

/* SignX - sign bit of X */
        vandnpd   %ymm4, %ymm2, %ymm7
        vmulpd    %ymm0, %ymm1, %ymm2

/* Poly = Poly*R + R */
        vfmadd213pd %ymm6, %ymm6, %ymm2
        vmovmskpd %ymm3, %ecx

/*
  RECONSTRUCTION:
  Final sign setting: Res = Poly^SignX
 */
        vxorpd    %ymm7, %ymm2, %ymm0
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
        vmovupd   %ymm4, 320(%rsp)
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

        call      JUMPTARGET(sin)

        vmovsd    %xmm0, 392(%rsp,%r15)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    320(%rsp,%r15), %xmm0
        vzeroupper

        call      JUMPTARGET(sin)

        vmovsd    %xmm0, 384(%rsp,%r15)
        jmp       .LBL_1_7

END (_ZGVdN4v_sin_avx2)

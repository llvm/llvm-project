/* Function sinf vectorized with AVX2.
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
#include "svml_s_trig_data.h"

	.text
ENTRY(_ZGVdN8v_sinf_avx2)
/*
   ALGORITHM DESCRIPTION:

   1) Range reduction to [-Pi/2; +Pi/2] interval
      a) Grab sign from source argument and save it.
      b) Remove sign using AND operation
      c) Getting octant Y by 1/Pi multiplication
      d) Add "Right Shifter" value
      e) Treat obtained value as integer for destination sign setting.
         Shift first bit of this value to the last (sign) position
      f) Change destination sign if source sign is negative
         using XOR operation.
      g) Subtract "Right Shifter" value
      h) Subtract Y*PI from X argument, where PI divided to 4 parts:
         X = X - Y*PI1 - Y*PI2 - Y*PI3 - Y*PI4;
   2) Polynomial (minimax for sin within [-Pi/2; +Pi/2] interval)
      a) Calculate X^2 = X * X
      b) Calculate polynomial:
         R = X + X * X^2 * (A3 + x^2 * (A5 + ......
   3) Destination sign setting
      a) Set shifted destination sign using XOR operation:
         R = XOR( R, S );
 */
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $448, %rsp
        movq      __svml_s_trig_data@GOTPCREL(%rip), %rax
        vmovdqa   %ymm0, %ymm5
        vmovups __sAbsMask(%rax), %ymm3
        vmovups __sInvPI(%rax), %ymm7
        vmovups __sRShifter(%rax), %ymm0
        vmovups __sPI1_FMA(%rax), %ymm1

/* b) Remove sign using AND operation */
        vandps    %ymm3, %ymm5, %ymm4

/*
  c) Getting octant Y by 1/Pi multiplication
  d) Add "Right Shifter" value
 */
        vfmadd213ps %ymm0, %ymm4, %ymm7

/* g) Subtract "Right Shifter" value */
        vsubps    %ymm0, %ymm7, %ymm2

/*
  e) Treat obtained value as integer for destination sign setting.
  Shift first bit of this value to the last (sign) position
 */
        vpslld    $31, %ymm7, %ymm6

/*
  h) Subtract Y*PI from X argument, where PI divided to 4 parts:
  X = X - Y*PI1 - Y*PI2 - Y*PI3;
 */
        vmovdqa   %ymm4, %ymm0
        vfnmadd231ps %ymm2, %ymm1, %ymm0

/* Check for large and special values */
        vcmpnle_uqps __sRangeReductionVal(%rax), %ymm4, %ymm4
        vfnmadd231ps __sPI2_FMA(%rax), %ymm2, %ymm0
        vfnmadd132ps __sPI3_FMA(%rax), %ymm0, %ymm2

/*
  2) Polynomial (minimax for sin within [-Pi/2; +Pi/2] interval)
  a) Calculate X^2 = X * X
  b) Calculate polynomial:
  R = X + X * X^2 * (A3 + x^2 * (A5 + ......
 */
        vmulps    %ymm2, %ymm2, %ymm1

/*
  f) Change destination sign if source sign is negative
  using XOR operation.
 */
        vandnps   %ymm5, %ymm3, %ymm0
        vxorps    %ymm6, %ymm2, %ymm3
        vmovups __sA9(%rax), %ymm2
        vfmadd213ps __sA7(%rax), %ymm1, %ymm2
        vfmadd213ps __sA5(%rax), %ymm1, %ymm2
        vfmadd213ps __sA3(%rax), %ymm1, %ymm2
        vmulps    %ymm1, %ymm2, %ymm6
        vfmadd213ps %ymm3, %ymm3, %ymm6
        vmovmskps %ymm4, %ecx

/*
  3) Destination sign setting
  a) Set shifted destination sign using XOR operation:
  R = XOR( R, S );
 */
        vxorps    %ymm0, %ymm6, %ymm0
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
        vmovups   %ymm5, 320(%rsp)
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

        call      JUMPTARGET(sinf)

        vmovss    %xmm0, 388(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        vmovss    320(%rsp,%r15,8), %xmm0
        vzeroupper

        call      JUMPTARGET(sinf)

        vmovss    %xmm0, 384(%rsp,%r15,8)
        jmp       .LBL_1_7

END(_ZGVdN8v_sinf_avx2)

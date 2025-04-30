/* Function cosf vectorized with SSE4.
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
ENTRY (_ZGVbN4v_cosf_sse4)
/*
  ALGORITHM DESCRIPTION:

   1) Range reduction to [-Pi/2; +Pi/2] interval
      a) We remove sign using AND operation
      b) Add Pi/2 value to argument X for Cos to Sin transformation
      c) Getting octant Y by 1/Pi multiplication
      d) Add "Right Shifter" value
      e) Treat obtained value as integer for destination sign setting.
         Shift first bit of this value to the last (sign) position
      f) Subtract "Right Shifter"  value
      g) Subtract 0.5 from result for octant correction
      h) Subtract Y*PI from X argument, where PI divided to 4 parts:
         X = X - Y*PI1 - Y*PI2 - Y*PI3 - Y*PI4;
   2) Polynomial (minimax for sin within [-Pi/2; +Pi/2] interval)
      a) Calculate X^2 = X * X
      b) Calculate polynomial:
         R = X + X * X^2 * (A3 + x^2 * (A5 + .....
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
        subq      $320, %rsp
        movaps    %xmm0, %xmm4
        movq      __svml_s_trig_data@GOTPCREL(%rip), %rax
        movups __sHalfPI(%rax), %xmm1
        movups __sRShifter(%rax), %xmm5

/* b) Add Pi/2 value to argument X for Cos to Sin transformation */
        addps     %xmm4, %xmm1

/*
  1) Range reduction to [-Pi/2; +Pi/2] interval
  c) Getting octant Y by 1/Pi multiplication
  d) Add "Right Shifter" (0x4B000000) value
 */
        mulps __sInvPI(%rax), %xmm1
        movups __sPI1(%rax), %xmm6
        addps     %xmm5, %xmm1

/*
  e) Treat obtained value as integer for destination sign setting.
  Shift first bit of this value to the last (sign) position (S << 31)
 */
        movaps    %xmm1, %xmm2

/* f) Subtract "Right Shifter" (0x4B000000) value */
        subps     %xmm5, %xmm1
        movups __sPI2(%rax), %xmm7
        pslld     $31, %xmm2
        movups __sPI3(%rax), %xmm5
        movups __sAbsMask(%rax), %xmm3

/* Check for large and special arguments */
        andps     %xmm4, %xmm3

/* g) Subtract 0.5 from result for octant correction */
        subps __sOneHalf(%rax), %xmm1
        cmpnleps __sRangeReductionVal(%rax), %xmm3

/*
  h) Subtract Y*PI from X argument, where PI divided to 4 parts:
  X = X - Y*PI1 - Y*PI2 - Y*PI3 - Y*PI4;
 */
        mulps     %xmm1, %xmm6
        mulps     %xmm1, %xmm7
        mulps     %xmm1, %xmm5
        subps     %xmm6, %xmm0
        movmskps  %xmm3, %ecx
        movups __sPI4(%rax), %xmm6
        subps     %xmm7, %xmm0
        mulps     %xmm6, %xmm1
        subps     %xmm5, %xmm0
        subps     %xmm1, %xmm0

/* a) Calculate X^2 = X * X */
        movaps    %xmm0, %xmm1
        mulps     %xmm0, %xmm1

/*
  3) Destination sign setting
  a) Set shifted destination sign using XOR operation:
  R = XOR( R, S );
 */
        xorps     %xmm2, %xmm0
        movups __sA9(%rax), %xmm2

/*
  b) Calculate polynomial:
  R = X + X * X^2 * (A3 + x^2 * (A5 + x^2 * (A7 + x^2 * (A9))));
 */
        mulps     %xmm1, %xmm2
        addps __sA7(%rax), %xmm2
        mulps     %xmm1, %xmm2
        addps __sA5(%rax), %xmm2
        mulps     %xmm1, %xmm2
        addps __sA3(%rax), %xmm2
        mulps     %xmm2, %xmm1
        mulps     %xmm0, %xmm1
        addps     %xmm1, %xmm0
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
        movups    %xmm4, 192(%rsp)
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

        call      JUMPTARGET(cosf)

        movss     %xmm0, 260(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        movss     192(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(cosf)

        movss     %xmm0, 256(%rsp,%r15,8)
        jmp       .LBL_1_7
END (_ZGVbN4v_cosf_sse4)

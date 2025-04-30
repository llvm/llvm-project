/* Function sincosf vectorized with SSE4.
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
ENTRY (_ZGVbN4vl4l4_sincosf_sse4)
/*
   ALGORITHM DESCRIPTION:

     1) Range reduction to [-Pi/4; +Pi/4] interval
        a) Grab sign from source argument and save it.
        b) Remove sign using AND operation
        c) Getting octant Y by 2/Pi multiplication
        d) Add "Right Shifter" value
        e) Treat obtained value as integer S for destination sign setting.
           SS = ((S-S&1)&2)<<30; For sin part
           SC = ((S+S&1)&2)<<30; For cos part
        f) Change destination sign if source sign is negative
           using XOR operation.
        g) Subtract "Right Shifter" (0x4B000000) value
        h) Subtract Y*(PI/2) from X argument, where PI/2 divided to 4 parts:
           X = X - Y*PI1 - Y*PI2 - Y*PI3 - Y*PI4;
     2) Polynomial (minimax for sin within  [-Pi/4; +Pi/4] interval)
        a) Calculate X^2 = X * X
        b) Calculate 2 polynomials for sin and cos:
           RS = X * ( A0 + X^2 * (A1 + x^2 * (A2 + x^2 * (A3))));
           RC = B0 + X^2 * (B1 + x^2 * (B2 + x^2 * (B3 + x^2 * (B4))));
        c) Swap RS & RC if first bit of obtained value after
           Right Shifting is set to 1. Using And, Andnot & Or operations.
     3) Destination sign setting
        a) Set shifted destination sign using XOR operation:
           R1 = XOR( RS, SS );
           R2 = XOR( RC, SC ).  */

        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $320, %rsp
        movq      __svml_s_trig_data@GOTPCREL(%rip), %rax
        movups    %xmm12, 176(%rsp)
        movups    %xmm9, 160(%rsp)
        movups __sAbsMask(%rax), %xmm12

/* Absolute argument computation */
        movaps    %xmm12, %xmm5
        andnps    %xmm0, %xmm12
        movups __sInvPI(%rax), %xmm7
        andps     %xmm0, %xmm5

/* c) Getting octant Y by 2/Pi multiplication
   d) Add "Right Shifter" value.  */
        mulps     %xmm5, %xmm7
        movups    %xmm10, 144(%rsp)
        movups __sPI1(%rax), %xmm10

/* h) Subtract Y*(PI/2) from X argument, where PI/2 divided to 3 parts:
      X = X - Y*PI1 - Y*PI2 - Y*PI3.  */
        movaps    %xmm10, %xmm1
        addps __sRShifter(%rax), %xmm7

/* e) Treat obtained value as integer S for destination sign setting */
        movaps    %xmm7, %xmm9

/* g) Subtract "Right Shifter" (0x4B000000) value */
        subps __sRShifter(%rax), %xmm7
        mulps     %xmm7, %xmm1
        pslld     $31, %xmm9
        movups __sPI2(%rax), %xmm6
        movups    %xmm13, 112(%rsp)
        movaps    %xmm5, %xmm13
        movaps    %xmm6, %xmm2
        subps     %xmm1, %xmm13
        mulps     %xmm7, %xmm2
        movups __sSignMask(%rax), %xmm3
        movaps    %xmm5, %xmm1
        movups __sOneHalf(%rax), %xmm4
        subps     %xmm2, %xmm13
        cmpnleps __sRangeReductionVal(%rax), %xmm5
        movaps    %xmm3, %xmm2
        andps     %xmm13, %xmm2
        xorps     %xmm2, %xmm4

/* Result sign calculations */
        xorps     %xmm2, %xmm3
        xorps     %xmm9, %xmm3

/* Add correction term 0.5 for cos() part */
        addps     %xmm7, %xmm4
        movmskps  %xmm5, %ecx
        mulps     %xmm4, %xmm10
        mulps     %xmm4, %xmm6
        subps     %xmm10, %xmm1
        movups __sPI3(%rax), %xmm10
        subps     %xmm6, %xmm1
        movaps    %xmm10, %xmm6
        mulps     %xmm7, %xmm6
        mulps     %xmm4, %xmm10
        subps     %xmm6, %xmm13
        subps     %xmm10, %xmm1
        movups __sPI4(%rax), %xmm6
        mulps     %xmm6, %xmm7
        mulps     %xmm6, %xmm4
        subps     %xmm7, %xmm13
        subps     %xmm4, %xmm1
        xorps     %xmm9, %xmm13
        xorps     %xmm3, %xmm1
        movaps    %xmm13, %xmm4
        movaps    %xmm1, %xmm2
        mulps     %xmm13, %xmm4
        mulps     %xmm1, %xmm2
        movups __sA9(%rax), %xmm7

/* 2) Polynomial (minimax for sin within  [-Pi/4; +Pi/4] interval)
      a) Calculate X^2 = X * X
      b) Calculate 2 polynomials for sin and cos:
         RS = X * ( A0 + X^2 * (A1 + x^2 * (A2 + x^2 * (A3))));
         RC = B0 + X^2 * (B1 + x^2 * (B2 + x^2 * (B3 + x^2 * (B4)))) */
        movaps    %xmm7, %xmm3
        mulps     %xmm4, %xmm3
        mulps     %xmm2, %xmm7
        addps __sA7(%rax), %xmm3
        addps __sA7(%rax), %xmm7
        mulps     %xmm4, %xmm3
        mulps     %xmm2, %xmm7
        addps __sA5(%rax), %xmm3
        addps __sA5(%rax), %xmm7
        mulps     %xmm4, %xmm3
        mulps     %xmm2, %xmm7
        addps __sA3(%rax), %xmm3
        addps __sA3(%rax), %xmm7
        mulps     %xmm3, %xmm4
        mulps     %xmm7, %xmm2
        mulps     %xmm13, %xmm4
        mulps     %xmm1, %xmm2
        addps     %xmm4, %xmm13
        addps     %xmm2, %xmm1
        xorps     %xmm12, %xmm13
        testl     %ecx, %ecx
        jne       .LBL_1_3

.LBL_1_2:
        cfi_remember_state
        movups    160(%rsp), %xmm9
        movaps    %xmm13, (%rdi)
        movups    144(%rsp), %xmm10
        movups    176(%rsp), %xmm12
        movups    112(%rsp), %xmm13
        movups    %xmm1, (%rsi)
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_1_3:
        cfi_restore_state
        movups    %xmm0, 128(%rsp)
        movups    %xmm13, 192(%rsp)
        movups    %xmm1, 256(%rsp)
        je        .LBL_1_2

        xorb      %dl, %dl
        xorl      %eax, %eax
        movups    %xmm8, 48(%rsp)
        movups    %xmm11, 32(%rsp)
        movups    %xmm14, 16(%rsp)
        movups    %xmm15, (%rsp)
        movq      %rsi, 64(%rsp)
        movq      %r12, 104(%rsp)
        cfi_offset_rel_rsp (12, 104)
        movb      %dl, %r12b
        movq      %r13, 96(%rsp)
        cfi_offset_rel_rsp (13, 96)
        movl      %eax, %r13d
        movq      %r14, 88(%rsp)
        cfi_offset_rel_rsp (14, 88)
        movl      %ecx, %r14d
        movq      %r15, 80(%rsp)
        cfi_offset_rel_rsp (15, 80)
        movq      %rbx, 72(%rsp)
        movq      %rdi, %rbx
        cfi_remember_state

.LBL_1_6:
        btl       %r13d, %r14d
        jc        .LBL_1_13

.LBL_1_7:
        lea       1(%r13), %esi
        btl       %esi, %r14d
        jc        .LBL_1_10

.LBL_1_8:
        incb      %r12b
        addl      $2, %r13d
        cmpb      $16, %r12b
        jb        .LBL_1_6

        movups    48(%rsp), %xmm8
        movq      %rbx, %rdi
        movups    32(%rsp), %xmm11
        movups    16(%rsp), %xmm14
        movups    (%rsp), %xmm15
        movq      64(%rsp), %rsi
        movq      104(%rsp), %r12
        cfi_restore (%r12)
        movq      96(%rsp), %r13
        cfi_restore (%r13)
        movq      88(%rsp), %r14
        cfi_restore (%r14)
        movq      80(%rsp), %r15
        cfi_restore (%r15)
        movq      72(%rsp), %rbx
        movups    192(%rsp), %xmm13
        movups    256(%rsp), %xmm1
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        movss     132(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(sinf)

        movss     %xmm0, 196(%rsp,%r15,8)
        movss     132(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(cosf)

        movss     %xmm0, 260(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_13:
        movzbl    %r12b, %r15d
        movss     128(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(sinf)

        movss     %xmm0, 192(%rsp,%r15,8)
        movss     128(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(cosf)

        movss     %xmm0, 256(%rsp,%r15,8)
        jmp       .LBL_1_7

END (_ZGVbN4vl4l4_sincosf_sse4)
libmvec_hidden_def(_ZGVbN4vl4l4_sincosf_sse4)

/* vvv version implemented with wrapper to vl4l4 variant.  */
ENTRY (_ZGVbN4vvv_sincosf_sse4)
#ifndef __ILP32__
        subq      $104, %rsp
        .cfi_def_cfa_offset 112
        movdqu    %xmm1, 32(%rsp)
        lea       (%rsp), %rdi
        movdqu    %xmm2, 48(%rdi)
        lea       16(%rsp), %rsi
        movdqu    %xmm3, 48(%rsi)
        movdqu    %xmm4, 64(%rsi)
        call      HIDDEN_JUMPTARGET(_ZGVbN4vl4l4_sincosf_sse4)
        movq      32(%rsp), %rdx
        movq      40(%rsp), %rsi
        movq      48(%rsp), %r8
        movq      56(%rsp), %r10
        movl      (%rsp), %eax
        movl      4(%rsp), %ecx
        movl      8(%rsp), %edi
        movl      12(%rsp), %r9d
        movl      %eax, (%rdx)
        movl      %ecx, (%rsi)
        movq      64(%rsp), %rax
        movq      72(%rsp), %rcx
        movl      %edi, (%r8)
        movl      %r9d, (%r10)
        movq      80(%rsp), %rdi
        movq      88(%rsp), %r9
        movl      16(%rsp), %r11d
        movl      20(%rsp), %edx
        movl      24(%rsp), %esi
        movl      28(%rsp), %r8d
        movl      %r11d, (%rax)
        movl      %edx, (%rcx)
        movl      %esi, (%rdi)
        movl      %r8d, (%r9)
        addq      $104, %rsp
        .cfi_def_cfa_offset 8
        ret
#else
        subl    $72, %esp
        .cfi_def_cfa_offset 80
        leal    48(%rsp), %esi
        movaps  %xmm1, 16(%esp)
        leal    32(%rsp), %edi
        movaps  %xmm2, (%esp)
        call    HIDDEN_JUMPTARGET(_ZGVbN4vl4l4_sincosf_sse4)
        movl    16(%esp), %eax
        movss   32(%esp), %xmm0
        movss   %xmm0, (%eax)
        movl    20(%esp), %eax
        movss   36(%esp), %xmm0
        movss   %xmm0, (%eax)
        movl    24(%esp), %eax
        movss   40(%esp), %xmm0
        movss   %xmm0, (%eax)
        movl    28(%esp), %eax
        movss   44(%esp), %xmm0
        movss   %xmm0, (%eax)
        movl    (%esp), %eax
        movss   48(%esp), %xmm0
        movss   %xmm0, (%eax)
        movl    4(%esp), %eax
        movss   52(%esp), %xmm0
        movss   %xmm0, (%eax)
        movl    8(%esp), %eax
        movss   56(%esp), %xmm0
        movss   %xmm0, (%eax)
        movl    12(%esp), %eax
        movss   60(%esp), %xmm0
        movss   %xmm0, (%eax)
        addl    $72, %esp
        .cfi_def_cfa_offset 8
        ret
#endif
END (_ZGVbN4vvv_sincosf_sse4)

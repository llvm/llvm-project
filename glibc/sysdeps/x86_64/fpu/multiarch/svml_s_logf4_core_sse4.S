/* Function logf vectorized with SSE4.
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
ENTRY (_ZGVbN4v_logf_sse4)
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
        subq      $320, %rsp

/* reduction: compute r,n */
        movaps    %xmm0, %xmm2

/* check for working range,
   set special argument mask (denormals/zero/Inf/NaN) */
        movq      __svml_slog_data@GOTPCREL(%rip), %rax
        movdqu _iHiDelta(%rax), %xmm1
        movdqu _iLoRange(%rax), %xmm4
        paddd     %xmm0, %xmm1
        movdqu _iBrkValue(%rax), %xmm3
        pcmpgtd   %xmm1, %xmm4
        movdqu _iOffExpoMask(%rax), %xmm1
        psubd     %xmm3, %xmm2
        pand      %xmm2, %xmm1

/* exponent_x (mantissa_x<4/3) or exponent_x+1 (mantissa_x>4/3) */
        psrad     $23, %xmm2
        paddd     %xmm3, %xmm1
        movups _sPoly_7(%rax), %xmm5

/* mantissa_x (mantissa_x<4/3), or 0.5*mantissa_x (mantissa_x>4/3) */
        cvtdq2ps  %xmm2, %xmm6

/* reduced argument R */
        subps _sOne(%rax), %xmm1
        movmskps  %xmm4, %ecx

/* final reconstruction:
   add exponent_value*log2 to polynomial result */
        mulps _sLn2(%rax), %xmm6

/* polynomial evaluation starts here */
        mulps     %xmm1, %xmm5
        addps _sPoly_6(%rax), %xmm5
        mulps     %xmm1, %xmm5
        addps _sPoly_5(%rax), %xmm5
        mulps     %xmm1, %xmm5
        addps _sPoly_4(%rax), %xmm5
        mulps     %xmm1, %xmm5
        addps _sPoly_3(%rax), %xmm5
        mulps     %xmm1, %xmm5
        addps _sPoly_2(%rax), %xmm5
        mulps     %xmm1, %xmm5
        addps _sPoly_1(%rax), %xmm5
        mulps     %xmm1, %xmm5

/* polynomial evaluation end */
        mulps     %xmm1, %xmm5
        addps     %xmm5, %xmm1
        addps     %xmm6, %xmm1
        testl     %ecx, %ecx
        jne       .LBL_1_3

.LBL_1_2:
        cfi_remember_state
        movdqa    %xmm1, %xmm0
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_1_3:
        cfi_restore_state
        movups    %xmm0, 192(%rsp)
        movups    %xmm1, 256(%rsp)
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
        movups    256(%rsp), %xmm1
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        movss     196(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(logf)

        movss     %xmm0, 260(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        movss     192(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(logf)

        movss     %xmm0, 256(%rsp,%r15,8)
        jmp       .LBL_1_7

END (_ZGVbN4v_logf_sse4)

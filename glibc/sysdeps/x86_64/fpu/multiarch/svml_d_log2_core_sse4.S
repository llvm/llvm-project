/* Function log vectorized with SSE4.
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
ENTRY (_ZGVbN2v_log_sse4)
/*
   ALGORITHM DESCRIPTION:

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
        subq      $320, %rsp
        movaps    %xmm0, %xmm6
        movq      __svml_dlog_data@GOTPCREL(%rip), %r8
        movaps    %xmm6, %xmm3
        movaps    %xmm6, %xmm2

/* isolate exponent bits */
        movaps    %xmm6, %xmm1
        psrlq     $20, %xmm1
        movups _ExpMask(%r8), %xmm5

/* preserve mantissa, set input exponent to 2^(-10) */
        andps     %xmm6, %xmm5
        orps _Two10(%r8), %xmm5

/* reciprocal approximation good to at least 11 bits */
        cvtpd2ps  %xmm5, %xmm7
        cmpltpd _MinNorm(%r8), %xmm3
        cmpnlepd _MaxNorm(%r8), %xmm2
        movlhps   %xmm7, %xmm7

/* combine and get argument value range mask */
        orps      %xmm2, %xmm3
        rcpps     %xmm7, %xmm0
        movmskpd  %xmm3, %eax
        movups _HalfMask(%r8), %xmm2

/* argument reduction started:  R = Mantissa*Rcp - 1 */
        andps     %xmm5, %xmm2
        cvtps2pd  %xmm0, %xmm4
        subpd     %xmm2, %xmm5

/* round reciprocal to nearest integer, will have 1+9 mantissa bits */
        roundpd   $0, %xmm4, %xmm4
        mulpd     %xmm4, %xmm2
        mulpd     %xmm4, %xmm5
        subpd _One(%r8), %xmm2
        addpd     %xmm2, %xmm5
        movups _Threshold(%r8), %xmm2

/* calculate index for table lookup */
        movaps    %xmm4, %xmm3
        cmpltpd   %xmm4, %xmm2
        pshufd    $221, %xmm1, %xmm7
        psrlq     $40, %xmm3

/* convert biased exponent to DP format */
        cvtdq2pd  %xmm7, %xmm0
        movd      %xmm3, %edx
        movups _poly_coeff_1(%r8), %xmm4

/* polynomial computation */
        mulpd     %xmm5, %xmm4
        andps _Bias(%r8), %xmm2
        orps _Bias1(%r8), %xmm2

/*
   Table stores -log(0.5*mantissa) for larger mantissas,
   adjust exponent accordingly
 */
        subpd     %xmm2, %xmm0
        addpd _poly_coeff_2(%r8), %xmm4

/* exponent*log(2.0) */
        mulpd _L2(%r8), %xmm0
        movaps    %xmm5, %xmm2
        mulpd     %xmm5, %xmm2
        movups _poly_coeff_3(%r8), %xmm7
        mulpd     %xmm5, %xmm7
        mulpd     %xmm2, %xmm4
        addpd _poly_coeff_4(%r8), %xmm7
        addpd     %xmm4, %xmm7
        mulpd     %xmm7, %xmm2
        movslq    %edx, %rdx
        pextrd    $2, %xmm3, %ecx

/*
   reconstruction:
   (exponent*log(2)) + (LogRcp + (R+poly))
 */
        addpd     %xmm2, %xmm5
        movslq    %ecx, %rcx
        movsd     _LogRcp_lookup(%r8,%rdx), %xmm1
        movhpd    _LogRcp_lookup(%r8,%rcx), %xmm1
        addpd     %xmm5, %xmm1
        addpd     %xmm1, %xmm0
        testl     %eax, %eax
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
        movups    %xmm6, 192(%rsp)
        movups    %xmm0, 256(%rsp)
        je        .LBL_1_2

        xorb      %cl, %cl
        xorl      %edx, %edx
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
        movb      %cl, %r12b
        movq      %r13, 160(%rsp)
        cfi_offset_rel_rsp (13, 160)
        movl      %eax, %r13d
        movq      %r14, 152(%rsp)
        cfi_offset_rel_rsp (14, 152)
        movl      %edx, %r14d
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
        shlq      $4, %r15
        movsd     200(%rsp,%r15), %xmm0

        call      JUMPTARGET(log)

        movsd     %xmm0, 264(%rsp,%r15)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        movsd     192(%rsp,%r15), %xmm0

        call      JUMPTARGET(log)

        movsd     %xmm0, 256(%rsp,%r15)
        jmp       .LBL_1_7

END (_ZGVbN2v_log_sse4)

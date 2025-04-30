/* Function sincos vectorized with SSE4.
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
ENTRY (_ZGVbN2vl8l8_sincos_sse4)
/*
   ALGORITHM DESCRIPTION:

     ( low accuracy ( < 4ulp ) or enhanced performance
      ( half of correct mantissa ) implementation )

     Argument representation:
     arg = N*Pi + R

     Result calculation:
     sin(arg) = sin(N*Pi + R) = (-1)^N * sin(R)
     arg + Pi/2 = (N'*Pi + R')
     cos(arg) = sin(arg+Pi/2) = sin(N'*Pi + R') = (-1)^N' * sin(R')
     sin(R), sin(R') are approximated by corresponding polynomial.  */

        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $320, %rsp
        movq      __svml_d_trig_data@GOTPCREL(%rip), %rax
        movups    %xmm11, 160(%rsp)
        movups    %xmm12, 144(%rsp)
        movups __dSignMask(%rax), %xmm11

/* ARGUMENT RANGE REDUCTION:
   Absolute argument: X' = |X| */
        movaps    %xmm11, %xmm4

/* Grab sign bit from argument */
        movaps    %xmm11, %xmm7
        movups __dInvPI(%rax), %xmm5
        andnps    %xmm0, %xmm4

/* SinY = X'*InvPi + RS : right shifter add */
        mulpd     %xmm4, %xmm5
        addpd __dRShifter(%rax), %xmm5

/* SinSignRes = Y<<63 : shift LSB to MSB place for result sign */
        movaps    %xmm5, %xmm12
        andps     %xmm0, %xmm7

/* SinN = Y - RS : right shifter sub */
        subpd __dRShifter(%rax), %xmm5
        movups    %xmm10, 176(%rsp)
        psllq     $63, %xmm12
        movups __dPI1(%rax), %xmm10

/* SinR = X' - SinN*Pi1 */
        movaps    %xmm10, %xmm1
        mulpd     %xmm5, %xmm1
        movups __dPI2(%rax), %xmm6

/* SinR = SinR - SinN*Pi1 */
        movaps    %xmm6, %xmm2
        mulpd     %xmm5, %xmm2
        movups    %xmm13, 112(%rsp)
        movaps    %xmm4, %xmm13
        subpd     %xmm1, %xmm13
        subpd     %xmm2, %xmm13

/* Sine result sign: SinRSign = SignMask & SinR */
        movaps    %xmm11, %xmm2

/* CosR = SinX - CosN*Pi1 */
        movaps    %xmm4, %xmm1
        movups __dOneHalf(%rax), %xmm3
        andps     %xmm13, %xmm2

/* Set SinRSign to 0.5 */
        orps      %xmm2, %xmm3

/* Update CosRSign and CosSignRes signs */
        xorps     %xmm11, %xmm2

/* CosN = SinN +(-)0.5 */
        addpd     %xmm5, %xmm3
        cmpnlepd __dRangeVal(%rax), %xmm4
        mulpd     %xmm3, %xmm10

/* CosR = CosR - CosN*Pi2 */
        mulpd     %xmm3, %xmm6
        subpd     %xmm10, %xmm1
        movmskpd  %xmm4, %ecx
        movups __dPI3(%rax), %xmm10
        xorps     %xmm12, %xmm2
        subpd     %xmm6, %xmm1

/* SinR = SinR - SinN*Pi3 */
        movaps    %xmm10, %xmm6

/* Final reconstruction.
   Combine Sin result's sign */
        xorps     %xmm7, %xmm12
        mulpd     %xmm5, %xmm6

/* CosR = CosR - CosN*Pi3 */
        mulpd     %xmm3, %xmm10
        subpd     %xmm6, %xmm13
        subpd     %xmm10, %xmm1
        movups __dPI4(%rax), %xmm6

/* SinR = SinR - SinN*Pi4 */
        mulpd     %xmm6, %xmm5

/* CosR = CosR - CosN*Pi4 */
        mulpd     %xmm6, %xmm3
        subpd     %xmm5, %xmm13
        subpd     %xmm3, %xmm1

/* SinR2 = SinR^2 */
        movaps    %xmm13, %xmm6

/* CosR2 = CosR^2 */
        movaps    %xmm1, %xmm10
        mulpd     %xmm13, %xmm6
        mulpd     %xmm1, %xmm10

/* Polynomial approximation */
        movups __dC7(%rax), %xmm5
        movaps    %xmm5, %xmm3
        mulpd     %xmm6, %xmm3
        mulpd     %xmm10, %xmm5
        addpd __dC6(%rax), %xmm3
        addpd __dC6(%rax), %xmm5
        mulpd     %xmm6, %xmm3
        mulpd     %xmm10, %xmm5
        addpd __dC5(%rax), %xmm3
        addpd __dC5(%rax), %xmm5
        mulpd     %xmm6, %xmm3
        mulpd     %xmm10, %xmm5
        addpd __dC4(%rax), %xmm3
        addpd __dC4(%rax), %xmm5

/* SinPoly = C3 + SinR2*(C4 + SinR2*(C5 + SinR2*(C6 + SinR2*C7))) */
        mulpd     %xmm6, %xmm3

/* CosPoly = C3 + CosR2*(C4 + CosR2*(C5 + CosR2*(C6 + CosR2*C7))) */
        mulpd     %xmm10, %xmm5
        addpd __dC3(%rax), %xmm3
        addpd __dC3(%rax), %xmm5

/* SinPoly = C2 + SinR2*SinPoly */
        mulpd     %xmm6, %xmm3

/* CosPoly = C2 + CosR2*CosPoly */
        mulpd     %xmm10, %xmm5
        addpd __dC2(%rax), %xmm3
        addpd __dC2(%rax), %xmm5

/* SinPoly = C1 + SinR2*SinPoly */
        mulpd     %xmm6, %xmm3

/* CosPoly = C1 + CosR2*CosPoly */
        mulpd     %xmm10, %xmm5
        addpd __dC1(%rax), %xmm3
        addpd __dC1(%rax), %xmm5

/* SinPoly = SinR2*SinPoly */
        mulpd     %xmm3, %xmm6

/* CosPoly = CosR2*CosPoly */
        mulpd     %xmm5, %xmm10

/* SinPoly = SinR*SinPoly */
        mulpd     %xmm13, %xmm6

/* CosPoly = CosR*CosPoly */
        mulpd     %xmm1, %xmm10
        addpd     %xmm6, %xmm13
        addpd     %xmm10, %xmm1

/* Update Sin result's sign */
        xorps     %xmm12, %xmm13

/* Update Cos result's sign */
        xorps     %xmm2, %xmm1
        testl     %ecx, %ecx
        jne       .LBL_1_3

.LBL_1_2:
        cfi_remember_state
        movups    176(%rsp), %xmm10
        movaps    %xmm13, (%rdi)
        movups    160(%rsp), %xmm11
        movups    144(%rsp), %xmm12
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
        movups    %xmm9, 32(%rsp)
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
        movups    32(%rsp), %xmm9
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
        shlq      $4, %r15
        movsd     136(%rsp,%r15), %xmm0

        call      JUMPTARGET(sin)

        movsd     %xmm0, 200(%rsp,%r15)
        movsd     136(%rsp,%r15), %xmm0

        call      JUMPTARGET(cos)

        movsd     %xmm0, 264(%rsp,%r15)
        jmp       .LBL_1_8

.LBL_1_13:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        movsd     128(%rsp,%r15), %xmm0

        call      JUMPTARGET(sin)

        movsd     %xmm0, 192(%rsp,%r15)
        movsd     128(%rsp,%r15), %xmm0

        call      JUMPTARGET(cos)

        movsd     %xmm0, 256(%rsp,%r15)
        jmp       .LBL_1_7
END (_ZGVbN2vl8l8_sincos_sse4)
libmvec_hidden_def(_ZGVbN2vl8l8_sincos_sse4)

/* vvv version implemented with wrapper to vl8l8 variant.  */
ENTRY (_ZGVbN2vvv_sincos_sse4)
#ifndef __ILP32__
        subq      $72, %rsp
        .cfi_def_cfa_offset 80
        movdqu    %xmm1, 32(%rsp)
        lea       (%rsp), %rdi
        movdqu    %xmm2, 48(%rdi)
        lea       16(%rsp), %rsi
        call      HIDDEN_JUMPTARGET(_ZGVbN2vl8l8_sincos_sse4)
        movq      32(%rsp), %rdx
        movq      48(%rsp), %rsi
        movq      40(%rsp), %r8
        movq      56(%rsp), %r10
        movq      (%rsp), %rax
        movq      16(%rsp), %rcx
        movq      8(%rsp), %rdi
        movq      24(%rsp), %r9
        movq      %rax, (%rdx)
        movq      %rcx, (%rsi)
        movq      %rdi, (%r8)
        movq      %r9, (%r10)
        addq      $72, %rsp
        .cfi_def_cfa_offset 8
        ret
#else
        subl    $72, %esp
        .cfi_def_cfa_offset 80
        leal    48(%rsp), %esi
        movaps  %xmm1, 16(%esp)
        leal    32(%rsp), %edi
        movaps  %xmm2, (%esp)
        call    HIDDEN_JUMPTARGET(_ZGVbN2vl8l8_sincos_sse4)
        movdqa  16(%esp), %xmm1
        movsd   32(%esp), %xmm0
        movq    %xmm1, %rax
        movdqa  (%esp), %xmm2
        movsd   %xmm0, (%eax)
        movsd   40(%esp), %xmm0
        pextrd  $1, %xmm1, %eax
        movsd   %xmm0, (%eax)
        movsd   48(%esp), %xmm0
        movq    %xmm2, %rax
        movsd   %xmm0, (%eax)
        movsd   56(%esp), %xmm0
        pextrd  $1, %xmm2, %eax
        movsd   %xmm0, (%eax)
        addl    $72, %esp
        .cfi_def_cfa_offset 8
        ret
#endif
END (_ZGVbN2vvv_sincos_sse4)

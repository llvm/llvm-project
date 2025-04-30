/* Function cos vectorized with SSE4.
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
ENTRY (_ZGVbN2v_cos_sse4)
/* ALGORITHM DESCRIPTION:

        ( low accuracy ( < 4ulp ) or enhanced performance
         ( half of correct mantissa ) implementation )

        Argument representation:
        arg + Pi/2 = (N*Pi + R)

        Result calculation:
        cos(arg) = sin(arg+Pi/2) = sin(N*Pi + R) = (-1)^N * sin(R)
        sin(R) is approximated by corresponding polynomial
 */
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $320, %rsp
        movaps    %xmm0, %xmm3
        movq      __svml_d_trig_data@GOTPCREL(%rip), %rax
        movups    __dHalfPI(%rax), %xmm2

/* ARGUMENT RANGE REDUCTION:
   Add Pi/2 to argument: X' = X+Pi/2
 */
        addpd     %xmm3, %xmm2
        movups    __dInvPI(%rax), %xmm5
        movups    __dAbsMask(%rax), %xmm4

/* Get absolute argument value: X' = |X'| */
        andps     %xmm2, %xmm4

/* Y = X'*InvPi + RS : right shifter add */
        mulpd     %xmm5, %xmm2

/* Check for large arguments path */
        cmpnlepd  __dRangeVal(%rax), %xmm4
        movups    __dRShifter(%rax), %xmm6
        addpd     %xmm6, %xmm2
        movmskpd  %xmm4, %ecx

/* N = Y - RS : right shifter sub */
        movaps    %xmm2, %xmm1

/* SignRes = Y<<63 : shift LSB to MSB place for result sign */
        psllq     $63, %xmm2
        subpd     %xmm6, %xmm1

/* N = N - 0.5 */
        subpd     __dOneHalf(%rax), %xmm1
        movups    __dPI1(%rax), %xmm7

/* R = X - N*Pi1 */
        mulpd     %xmm1, %xmm7
        movups    __dPI2(%rax), %xmm4

/* R = R - N*Pi2 */
        mulpd     %xmm1, %xmm4
        subpd     %xmm7, %xmm0
        movups    __dPI3(%rax), %xmm5

/* R = R - N*Pi3 */
        mulpd     %xmm1, %xmm5
        subpd     %xmm4, %xmm0

/* R = R - N*Pi4 */
        movups     __dPI4(%rax), %xmm6
        mulpd     %xmm6, %xmm1
        subpd     %xmm5, %xmm0
        subpd     %xmm1, %xmm0

/* POLYNOMIAL APPROXIMATION: R2 = R*R */
        movaps    %xmm0, %xmm4
        mulpd     %xmm0, %xmm4
        movups    __dC7(%rax), %xmm1
        mulpd     %xmm4, %xmm1
        addpd     __dC6(%rax), %xmm1
        mulpd     %xmm4, %xmm1
        addpd     __dC5(%rax), %xmm1
        mulpd     %xmm4, %xmm1
        addpd     __dC4(%rax), %xmm1

/* Poly = C3+R2*(C4+R2*(C5+R2*(C6+R2*C7))) */
        mulpd     %xmm4, %xmm1
        addpd     __dC3(%rax), %xmm1

/* Poly = R+R*(R2*(C1+R2*(C2+R2*Poly))) */
        mulpd     %xmm4, %xmm1
        addpd     __dC2(%rax), %xmm1
        mulpd     %xmm4, %xmm1
        addpd     __dC1(%rax), %xmm1
        mulpd     %xmm1, %xmm4
        mulpd     %xmm0, %xmm4
        addpd     %xmm4, %xmm0

/* RECONSTRUCTION:
   Final sign setting: Res = Poly^SignRes */
        xorps     %xmm2, %xmm0
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
        movups    %xmm3, 192(%rsp)
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
        shlq      $4, %r15
        movsd     200(%rsp,%r15), %xmm0

        call      JUMPTARGET(cos)

        movsd     %xmm0, 264(%rsp,%r15)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        movsd     192(%rsp,%r15), %xmm0

        call      JUMPTARGET(cos)

        movsd     %xmm0, 256(%rsp,%r15)
        jmp       .LBL_1_7

END (_ZGVbN2v_cos_sse4)

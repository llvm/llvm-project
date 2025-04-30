/* Function sincos vectorized with AVX2.
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
ENTRY (_ZGVdN4vl8l8_sincos_avx2)
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
        subq      $448, %rsp
        movq      __svml_d_trig_data@GOTPCREL(%rip), %rax
        vmovups   %ymm14, 288(%rsp)
        vmovups   %ymm8, 352(%rsp)
        vmovupd __dSignMask(%rax), %ymm6
        vmovupd __dInvPI(%rax), %ymm2
        vmovupd __dPI1_FMA(%rax), %ymm5
        vmovups   %ymm9, 224(%rsp)

/* ARGUMENT RANGE REDUCTION:
   Absolute argument: X' = |X| */
        vandnpd   %ymm0, %ymm6, %ymm1

/* SinY = X'*InvPi + RS : right shifter add */
        vfmadd213pd __dRShifter(%rax), %ymm1, %ymm2

/* SinSignRes = Y<<63 : shift LSB to MSB place for result sign */
        vpsllq    $63, %ymm2, %ymm4

/* SinN = Y - RS : right shifter sub */
        vsubpd __dRShifter(%rax), %ymm2, %ymm2

/* SinR = X' - SinN*Pi1 */
        vmovdqa   %ymm1, %ymm14
        vfnmadd231pd %ymm2, %ymm5, %ymm14

/* SinR = SinR - SinN*Pi1 */
        vfnmadd231pd __dPI2_FMA(%rax), %ymm2, %ymm14

/* Sine result sign: SinRSign = SignMask & SinR */
        vandpd    %ymm14, %ymm6, %ymm7

/* Set SinRSign to 0.5 */
        vorpd __dOneHalf(%rax), %ymm7, %ymm3

/* CosN = SinN +(-)0.5 */
        vaddpd    %ymm3, %ymm2, %ymm3

/* CosR = SinX - CosN*Pi1 */
        vmovdqa   %ymm1, %ymm8
        vfnmadd231pd %ymm3, %ymm5, %ymm8
        vmovupd __dPI3_FMA(%rax), %ymm5
        vcmpnle_uqpd __dRangeVal(%rax), %ymm1, %ymm1

/* CosR = CosR - CosN*Pi2 */
        vfnmadd231pd __dPI2_FMA(%rax), %ymm3, %ymm8

/* SinR = SinR - SinN*Pi3 */
        vfnmadd213pd %ymm14, %ymm5, %ymm2

/* CosR = CosR - CosN*Pi3 */
        vfnmadd213pd %ymm8, %ymm5, %ymm3
        vmovupd __dC6(%rax), %ymm8

/* SinR2 = SinR^2 */
        vmulpd    %ymm2, %ymm2, %ymm14

/* CosR2 = CosR^2 */
        vmulpd    %ymm3, %ymm3, %ymm5

/* Grab SignX */
        vandpd    %ymm0, %ymm6, %ymm9

/* Update CosRSign and CosSignRes signs */
        vxorpd    %ymm6, %ymm7, %ymm6
        vxorpd    %ymm6, %ymm4, %ymm7

/* Update sign SinSignRes */
        vxorpd    %ymm9, %ymm4, %ymm6

/* Polynomial approximation */
        vmovupd __dC7(%rax), %ymm4
        vmovdqa   %ymm8, %ymm9
        vfmadd231pd __dC7(%rax), %ymm14, %ymm9
        vfmadd213pd %ymm8, %ymm5, %ymm4
        vfmadd213pd __dC5(%rax), %ymm14, %ymm9
        vfmadd213pd __dC5(%rax), %ymm5, %ymm4
        vfmadd213pd __dC4(%rax), %ymm14, %ymm9
        vfmadd213pd __dC4(%rax), %ymm5, %ymm4

/* SinPoly = C3 + SinR2*(C4 + SinR2*(C5 + SinR2*(C6 + SinR2*C7))) */
        vfmadd213pd __dC3(%rax), %ymm14, %ymm9

/* CosPoly = C3 + CosR2*(C4 + CosR2*(C5 + CosR2*(C6 + CosR2*C7))) */
        vfmadd213pd __dC3(%rax), %ymm5, %ymm4

/* SinPoly = C2 + SinR2*SinPoly */
        vfmadd213pd __dC2(%rax), %ymm14, %ymm9

/* CosPoly = C2 + CosR2*CosPoly */
        vfmadd213pd __dC2(%rax), %ymm5, %ymm4

/* SinPoly = C1 + SinR2*SinPoly */
        vfmadd213pd __dC1(%rax), %ymm14, %ymm9

/* CosPoly = C1 + CosR2*CosPoly */
        vfmadd213pd __dC1(%rax), %ymm5, %ymm4

/* SinPoly = SinR2*SinPoly */
        vmulpd    %ymm14, %ymm9, %ymm8

/* CosPoly = CosR2*CosPoly */
        vmulpd    %ymm5, %ymm4, %ymm4

/* SinPoly = SinR*SinPoly */
        vfmadd213pd %ymm2, %ymm2, %ymm8

/* CosPoly = CosR*CosPoly */
        vfmadd213pd %ymm3, %ymm3, %ymm4
        vmovmskpd %ymm1, %ecx

/* Final reconstruction
   Update Sin result's sign */
        vxorpd    %ymm6, %ymm8, %ymm3

/* Update Cos result's sign */
        vxorpd    %ymm7, %ymm4, %ymm2
        testl     %ecx, %ecx
        jne       .LBL_1_3

.LBL_1_2:
        cfi_remember_state
        vmovups   352(%rsp), %ymm8
        vmovups   224(%rsp), %ymm9
        vmovups   288(%rsp), %ymm14
        vmovupd   %ymm2, (%rsi)
        vmovdqa   %ymm3, (%rdi)
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_1_3:
        cfi_restore_state
        vmovupd   %ymm0, 256(%rsp)
        vmovupd   %ymm3, 320(%rsp)
        vmovupd   %ymm2, 384(%rsp)
        je        .LBL_1_2

        xorb      %dl, %dl
        xorl      %eax, %eax
        vmovups   %ymm10, 128(%rsp)
        vmovups   %ymm11, 96(%rsp)
        vmovups   %ymm12, 64(%rsp)
        vmovups   %ymm13, 32(%rsp)
        vmovups   %ymm15, (%rsp)
        movq      %rsi, 160(%rsp)
        movq      %r12, 200(%rsp)
        cfi_offset_rel_rsp (12, 200)
        movb      %dl, %r12b
        movq      %r13, 192(%rsp)
        cfi_offset_rel_rsp (13, 192)
        movl      %eax, %r13d
        movq      %r14, 184(%rsp)
        cfi_offset_rel_rsp (14, 184)
        movl      %ecx, %r14d
        movq      %r15, 176(%rsp)
        cfi_offset_rel_rsp (15, 176)
        movq      %rbx, 168(%rsp)
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

        vmovups   128(%rsp), %ymm10
        movq      %rbx, %rdi
        vmovups   96(%rsp), %ymm11
        vmovups   64(%rsp), %ymm12
        vmovups   32(%rsp), %ymm13
        vmovups   (%rsp), %ymm15
        vmovupd   320(%rsp), %ymm3
        vmovupd   384(%rsp), %ymm2
        movq      160(%rsp), %rsi
        movq      200(%rsp), %r12
        cfi_restore (%r12)
        movq      192(%rsp), %r13
        cfi_restore (%r13)
        movq      184(%rsp), %r14
        cfi_restore (%r14)
        movq      176(%rsp), %r15
        cfi_restore (%r15)
        movq      168(%rsp), %rbx
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    264(%rsp,%r15), %xmm0
        vzeroupper

        call      JUMPTARGET(sin)

        vmovsd    %xmm0, 328(%rsp,%r15)
        vmovsd    264(%rsp,%r15), %xmm0

        call      JUMPTARGET(cos)

        vmovsd    %xmm0, 392(%rsp,%r15)
        jmp       .LBL_1_8

.LBL_1_13:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    256(%rsp,%r15), %xmm0
        vzeroupper

        call      JUMPTARGET(sin)

        vmovsd    %xmm0, 320(%rsp,%r15)
        vmovsd    256(%rsp,%r15), %xmm0

        call      JUMPTARGET(cos)

        vmovsd    %xmm0, 384(%rsp,%r15)
        jmp       .LBL_1_7

END (_ZGVdN4vl8l8_sincos_avx2)
libmvec_hidden_def(_ZGVdN4vl8l8_sincos_avx2)

/* vvv version implemented with wrapper to vl8l8 variant.  */
ENTRY (_ZGVdN4vvv_sincos_avx2)
#ifndef __ILP32__
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-32, %rsp
        subq      $128, %rsp
        vmovdqu   %ymm1, 64(%rsp)
        lea       (%rsp), %rdi
        vmovdqu   %ymm2, 96(%rdi)
        lea       32(%rsp), %rsi
        call      HIDDEN_JUMPTARGET(_ZGVdN4vl8l8_sincos_avx2)
        movq      64(%rsp), %rdx
        movq      96(%rsp), %rsi
        movq      72(%rsp), %r8
        movq      104(%rsp), %r10
        movq      (%rsp), %rax
        movq      32(%rsp), %rcx
        movq      8(%rsp), %rdi
        movq      40(%rsp), %r9
        movq      %rax, (%rdx)
        movq      %rcx, (%rsi)
        movq      80(%rsp), %rax
        movq      112(%rsp), %rcx
        movq      %rdi, (%r8)
        movq      %r9, (%r10)
        movq      88(%rsp), %rdi
        movq      120(%rsp), %r9
        movq      16(%rsp), %r11
        movq      48(%rsp), %rdx
        movq      24(%rsp), %rsi
        movq      56(%rsp), %r8
        movq      %r11, (%rax)
        movq      %rdx, (%rcx)
        movq      %rsi, (%rdi)
        movq      %r8, (%r9)
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret
#else
        leal    8(%rsp), %r10d
        .cfi_def_cfa 10, 0
        andl    $-32, %esp
        pushq   -8(%r10d)
        pushq   %rbp
        .cfi_escape 0x10,0x6,0x2,0x76,0
        movl    %esp, %ebp
        pushq   %r10
        .cfi_escape 0xf,0x3,0x76,0x78,0x6
        leal    -48(%rbp), %esi
        leal    -80(%rbp), %edi
        subl    $104, %esp
        vmovaps %xmm1, -96(%ebp)
        vmovaps %xmm2, -112(%ebp)
        call    HIDDEN_JUMPTARGET(_ZGVdN4vl8l8_sincos_avx2)
        movl    -96(%ebp), %eax
        vmovsd  -80(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -92(%ebp), %eax
        vmovsd  -72(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -88(%ebp), %eax
        vmovsd  -64(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -84(%ebp), %eax
        vmovsd  -56(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -112(%ebp), %eax
        vmovsd  -48(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -108(%ebp), %eax
        vmovsd  -40(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -104(%ebp), %eax
        vmovsd  -32(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -100(%ebp), %eax
        vmovsd  -24(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        addl    $104, %esp
        popq    %r10
        .cfi_def_cfa 10, 0
        popq    %rbp
        leal    -8(%r10), %esp
        .cfi_def_cfa 7, 8
        ret
#endif
END (_ZGVdN4vvv_sincos_avx2)

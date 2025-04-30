/* Function sincos vectorized with AVX-512. KNL and SKX versions.
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
#include "svml_d_wrapper_impl.h"

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

	.text
ENTRY (_ZGVeN8vl8l8_sincos_knl)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512_fFF _ZGVdN4vl8l8_sincos
#else
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $1344, %rsp
        movq      __svml_d_trig_data@GOTPCREL(%rip), %rax
        vmovaps   %zmm0, %zmm4
        movq      $-1, %rdx
        vmovups __dSignMask(%rax), %zmm12
        vmovups __dInvPI(%rax), %zmm5

/* ARGUMENT RANGE REDUCTION:
   Absolute argument: X' = |X| */
        vpandnq   %zmm4, %zmm12, %zmm3
        vmovups __dPI1_FMA(%rax), %zmm7
        vmovups __dPI3_FMA(%rax), %zmm9

/* SinR = X' - SinN*Pi1 */
        vmovaps   %zmm3, %zmm8

/* CosR = SinX - CosN*Pi1 */
        vmovaps   %zmm3, %zmm10

/* SinY = X'*InvPi + RS : right shifter add */
        vfmadd213pd __dRShifter(%rax), %zmm3, %zmm5
        vmovups __dC6(%rax), %zmm13

/* SinN = Y - RS : right shifter sub */
        vsubpd __dRShifter(%rax), %zmm5, %zmm1
        vmovaps   %zmm13, %zmm14

/* SinSignRes = Y<<63 : shift LSB to MSB place for result sign */
        vpsllq    $63, %zmm5, %zmm2
        vcmppd    $22, __dRangeVal(%rax), %zmm3, %k1

/* Update CosRSign and CosSignRes signs */
        vmovaps   %zmm12, %zmm5
        vfnmadd231pd %zmm1, %zmm7, %zmm8

/* SinR = SinR - SinN*Pi1 */
        vfnmadd231pd __dPI2_FMA(%rax), %zmm1, %zmm8

/* Sine result sign: SinRSign = SignMask & SinR */
        vpandq    %zmm8, %zmm12, %zmm11

/* Set SinRSign to 0.5 */
        vporq __dOneHalf(%rax), %zmm11, %zmm6
        vpternlogq $150, %zmm2, %zmm11, %zmm5

/* Update sign SinSignRes */
        vpternlogq $120, %zmm4, %zmm12, %zmm2

/* Polynomial approximation */
        vmovups __dC7(%rax), %zmm11

/* CosN = SinN +(-)0.5 */
        vaddpd    %zmm6, %zmm1, %zmm0

/* SinR = SinR - SinN*Pi3 */
        vfnmadd213pd %zmm8, %zmm9, %zmm1
        vfnmadd231pd %zmm0, %zmm7, %zmm10

/* SinR2 = SinR^2 */
        vmulpd    %zmm1, %zmm1, %zmm15

/* Grab SignX
   CosR = CosR - CosN*Pi2 */
        vfnmadd231pd __dPI2_FMA(%rax), %zmm0, %zmm10
        vfmadd231pd __dC7(%rax), %zmm15, %zmm14

/* CosR = CosR - CosN*Pi3 */
        vfnmadd213pd %zmm10, %zmm9, %zmm0
        vfmadd213pd __dC5(%rax), %zmm15, %zmm14

/* CosR2 = CosR^2 */
        vmulpd    %zmm0, %zmm0, %zmm12
        vfmadd213pd __dC4(%rax), %zmm15, %zmm14
        vfmadd213pd %zmm13, %zmm12, %zmm11

/* SinPoly = C3 + SinR2*(C4 + SinR2*(C5 + SinR2*(C6 + SinR2*C7))) */
        vfmadd213pd __dC3(%rax), %zmm15, %zmm14
        vfmadd213pd __dC5(%rax), %zmm12, %zmm11

/* SinPoly = C2 + SinR2*SinPoly */
        vfmadd213pd __dC2(%rax), %zmm15, %zmm14
        vfmadd213pd __dC4(%rax), %zmm12, %zmm11

/* SinPoly = C1 + SinR2*SinPoly */
        vfmadd213pd __dC1(%rax), %zmm15, %zmm14

/* CosPoly = C3 + CosR2*(C4 + CosR2*(C5 + CosR2*(C6 + CosR2*C7))) */
        vfmadd213pd __dC3(%rax), %zmm12, %zmm11

/* SinPoly = SinR2*SinPoly */
        vmulpd    %zmm15, %zmm14, %zmm13

/* CosPoly = C2 + CosR2*CosPoly */
        vfmadd213pd __dC2(%rax), %zmm12, %zmm11

/* SinPoly = SinR*SinPoly */
        vfmadd213pd %zmm1, %zmm1, %zmm13
        vpbroadcastq %rdx, %zmm1{%k1}{z}

/* CosPoly = C1 + CosR2*CosPoly */
        vfmadd213pd __dC1(%rax), %zmm12, %zmm11
        vptestmq  %zmm1, %zmm1, %k0
        kmovw     %k0, %ecx

/* CosPoly = CosR2*CosPoly */
        vmulpd    %zmm12, %zmm11, %zmm14
        movzbl    %cl, %ecx

/* CosPoly = CosR*CosPoly */
        vfmadd213pd %zmm0, %zmm0, %zmm14

/* Final reconstruction.
   Update Sin result's sign */
        vpxorq    %zmm2, %zmm13, %zmm0

/* Update Cos result's sign */
        vpxorq    %zmm5, %zmm14, %zmm2
        testl     %ecx, %ecx
        jne       .LBL_1_3

.LBL_1_2:
        cfi_remember_state
        vmovups   %zmm0, (%rdi)
        vmovups   %zmm2, (%rsi)
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_1_3:
        cfi_restore_state
        vmovups   %zmm4, 1152(%rsp)
        vmovups   %zmm0, 1216(%rsp)
        vmovups   %zmm2, 1280(%rsp)
        je        .LBL_1_2

        xorb      %dl, %dl
        kmovw     %k4, 1048(%rsp)
        xorl      %eax, %eax
        kmovw     %k5, 1040(%rsp)
        kmovw     %k6, 1032(%rsp)
        kmovw     %k7, 1024(%rsp)
        vmovups   %zmm16, 960(%rsp)
        vmovups   %zmm17, 896(%rsp)
        vmovups   %zmm18, 832(%rsp)
        vmovups   %zmm19, 768(%rsp)
        vmovups   %zmm20, 704(%rsp)
        vmovups   %zmm21, 640(%rsp)
        vmovups   %zmm22, 576(%rsp)
        vmovups   %zmm23, 512(%rsp)
        vmovups   %zmm24, 448(%rsp)
        vmovups   %zmm25, 384(%rsp)
        vmovups   %zmm26, 320(%rsp)
        vmovups   %zmm27, 256(%rsp)
        vmovups   %zmm28, 192(%rsp)
        vmovups   %zmm29, 128(%rsp)
        vmovups   %zmm30, 64(%rsp)
        vmovups   %zmm31, (%rsp)
        movq      %rsi, 1056(%rsp)
        movq      %r12, 1096(%rsp)
        cfi_offset_rel_rsp (12, 1096)
        movb      %dl, %r12b
        movq      %r13, 1088(%rsp)
        cfi_offset_rel_rsp (13, 1088)
        movl      %eax, %r13d
        movq      %r14, 1080(%rsp)
        cfi_offset_rel_rsp (14, 1080)
        movl      %ecx, %r14d
        movq      %r15, 1072(%rsp)
        cfi_offset_rel_rsp (15, 1072)
        movq      %rbx, 1064(%rsp)
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
        addb      $1, %r12b
        addl      $2, %r13d
        cmpb      $16, %r12b
        jb        .LBL_1_6

        movq      %rbx, %rdi
        kmovw     1048(%rsp), %k4
        movq      1056(%rsp), %rsi
        kmovw     1040(%rsp), %k5
        movq      1096(%rsp), %r12
        cfi_restore (%r12)
        kmovw     1032(%rsp), %k6
        movq      1088(%rsp), %r13
        cfi_restore (%r13)
        kmovw     1024(%rsp), %k7
        vmovups   960(%rsp), %zmm16
        vmovups   896(%rsp), %zmm17
        vmovups   832(%rsp), %zmm18
        vmovups   768(%rsp), %zmm19
        vmovups   704(%rsp), %zmm20
        vmovups   640(%rsp), %zmm21
        vmovups   576(%rsp), %zmm22
        vmovups   512(%rsp), %zmm23
        vmovups   448(%rsp), %zmm24
        vmovups   384(%rsp), %zmm25
        vmovups   320(%rsp), %zmm26
        vmovups   256(%rsp), %zmm27
        vmovups   192(%rsp), %zmm28
        vmovups   128(%rsp), %zmm29
        vmovups   64(%rsp), %zmm30
        vmovups   (%rsp), %zmm31
        movq      1080(%rsp), %r14
        cfi_restore (%r14)
        movq      1072(%rsp), %r15
        cfi_restore (%r15)
        movq      1064(%rsp), %rbx
        vmovups   1216(%rsp), %zmm0
        vmovups   1280(%rsp), %zmm2
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    1160(%rsp,%r15), %xmm0

        call      JUMPTARGET(sin)

        vmovsd    %xmm0, 1224(%rsp,%r15)
        vmovsd    1160(%rsp,%r15), %xmm0

        call      JUMPTARGET(cos)

        vmovsd    %xmm0, 1288(%rsp,%r15)
        jmp       .LBL_1_8

.LBL_1_13:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    1152(%rsp,%r15), %xmm0

        call      JUMPTARGET(sin)

        vmovsd    %xmm0, 1216(%rsp,%r15)
        vmovsd    1152(%rsp,%r15), %xmm0

        call      JUMPTARGET(cos)

        vmovsd    %xmm0, 1280(%rsp,%r15)
        jmp       .LBL_1_7

#endif
END (_ZGVeN8vl8l8_sincos_knl)
libmvec_hidden_def(_ZGVeN8vl8l8_sincos_knl)

ENTRY (_ZGVeN8vl8l8_sincos_skx)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512_fFF _ZGVdN4vl8l8_sincos
#else
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $1344, %rsp
        movq      __svml_d_trig_data@GOTPCREL(%rip), %rax
        vmovaps   %zmm0, %zmm8
        vmovups __dSignMask(%rax), %zmm4
        vmovups __dInvPI(%rax), %zmm9
        vmovups __dRShifter(%rax), %zmm10
        vmovups __dPI1_FMA(%rax), %zmm13
        vmovups __dPI2_FMA(%rax), %zmm14
        vmovups __dOneHalf(%rax), %zmm11
        vmovups __dPI3_FMA(%rax), %zmm2

/* ARGUMENT RANGE REDUCTION:
   Absolute argument: X' = |X| */
        vandnpd   %zmm8, %zmm4, %zmm7

/* SinY = X'*InvPi + RS : right shifter add */
        vfmadd213pd %zmm10, %zmm7, %zmm9
        vcmppd    $18, __dRangeVal(%rax), %zmm7, %k1

/* SinSignRes = Y<<63 : shift LSB to MSB place for result sign */
        vpsllq    $63, %zmm9, %zmm6

/* SinN = Y - RS : right shifter sub */
        vsubpd    %zmm10, %zmm9, %zmm5
        vmovups __dC5(%rax), %zmm9
        vmovups __dC4(%rax), %zmm10

/* SinR = X' - SinN*Pi1 */
        vmovaps   %zmm7, %zmm15
        vfnmadd231pd %zmm5, %zmm13, %zmm15

/* SinR = SinR - SinN*Pi1 */
        vfnmadd231pd %zmm5, %zmm14, %zmm15

/* Sine result sign: SinRSign = SignMask & SinR */
        vandpd    %zmm15, %zmm4, %zmm1

/* Set SinRSign to 0.5 */
        vorpd     %zmm1, %zmm11, %zmm12
        vmovups __dC3(%rax), %zmm11

/* CosN = SinN +(-)0.5 */
        vaddpd    %zmm12, %zmm5, %zmm3

/* SinR = SinR - SinN*Pi3 */
        vfnmadd213pd %zmm15, %zmm2, %zmm5
        vmovups __dC2(%rax), %zmm12

/* SinR2 = SinR^2 */
        vmulpd    %zmm5, %zmm5, %zmm15

/* CosR = SinX - CosN*Pi1 */
        vmovaps   %zmm7, %zmm0
        vfnmadd231pd %zmm3, %zmm13, %zmm0
        vmovups __dC1(%rax), %zmm13

/* Grab SignX
   CosR = CosR - CosN*Pi2 */
        vfnmadd231pd %zmm3, %zmm14, %zmm0

/* CosR = CosR - CosN*Pi3 */
        vfnmadd213pd %zmm0, %zmm2, %zmm3

/* Polynomial approximation */
        vmovups __dC7(%rax), %zmm0

/* Update CosRSign and CosSignRes signs */
        vmovaps   %zmm4, %zmm2
        vpternlogq $150, %zmm6, %zmm1, %zmm2

/* Update sign SinSignRes */
        vpternlogq $120, %zmm8, %zmm4, %zmm6

/* CosR2 = CosR^2 */
        vmulpd    %zmm3, %zmm3, %zmm1
        vmovups __dC6(%rax), %zmm4
        vmovaps   %zmm0, %zmm14
        vfmadd213pd %zmm4, %zmm1, %zmm0
        vfmadd213pd %zmm4, %zmm15, %zmm14
        vfmadd213pd %zmm9, %zmm1, %zmm0
        vfmadd213pd %zmm9, %zmm15, %zmm14
        vfmadd213pd %zmm10, %zmm1, %zmm0
        vfmadd213pd %zmm10, %zmm15, %zmm14

/* CosPoly = C3 + CosR2*(C4 + CosR2*(C5 + CosR2*(C6 + CosR2*C7))) */
        vfmadd213pd %zmm11, %zmm1, %zmm0

/* SinPoly = C3 + SinR2*(C4 + SinR2*(C5 + SinR2*(C6 + SinR2*C7))) */
        vfmadd213pd %zmm11, %zmm15, %zmm14

/* CosPoly = C2 + CosR2*CosPoly */
        vfmadd213pd %zmm12, %zmm1, %zmm0

/* SinPoly = C2 + SinR2*SinPoly */
        vfmadd213pd %zmm12, %zmm15, %zmm14

/* CosPoly = C1 + CosR2*CosPoly */
        vfmadd213pd %zmm13, %zmm1, %zmm0

/* SinPoly = C1 + SinR2*SinPoly */
        vfmadd213pd %zmm13, %zmm15, %zmm14

/* CosPoly = CosR2*CosPoly */
        vmulpd    %zmm1, %zmm0, %zmm1

/* SinPoly = SinR2*SinPoly */
        vmulpd    %zmm15, %zmm14, %zmm4

/* CosPoly = CosR*CosPoly */
        vfmadd213pd %zmm3, %zmm3, %zmm1

/* SinPoly = SinR*SinPoly */
        vfmadd213pd %zmm5, %zmm5, %zmm4
        vpbroadcastq .L_2il0floatpacket.15(%rip), %zmm3

/* Update Cos result's sign */
        vxorpd    %zmm2, %zmm1, %zmm1

/* Final reconstruction.
   Update Sin result's sign */
        vxorpd    %zmm6, %zmm4, %zmm0
        vpandnq   %zmm7, %zmm7, %zmm3{%k1}
        vcmppd    $3, %zmm3, %zmm3, %k0
        kmovw     %k0, %ecx
        testl     %ecx, %ecx
        jne       .LBL_2_3

.LBL_2_2:
        cfi_remember_state
        vmovups   %zmm0, (%rdi)
        vmovups   %zmm1, (%rsi)
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_2_3:
        cfi_restore_state
        vmovups   %zmm8, 1152(%rsp)
        vmovups   %zmm0, 1216(%rsp)
        vmovups   %zmm1, 1280(%rsp)
        je        .LBL_2_2

        xorb      %dl, %dl
        xorl      %eax, %eax
        kmovw     %k4, 1048(%rsp)
        kmovw     %k5, 1040(%rsp)
        kmovw     %k6, 1032(%rsp)
        kmovw     %k7, 1024(%rsp)
        vmovups   %zmm16, 960(%rsp)
        vmovups   %zmm17, 896(%rsp)
        vmovups   %zmm18, 832(%rsp)
        vmovups   %zmm19, 768(%rsp)
        vmovups   %zmm20, 704(%rsp)
        vmovups   %zmm21, 640(%rsp)
        vmovups   %zmm22, 576(%rsp)
        vmovups   %zmm23, 512(%rsp)
        vmovups   %zmm24, 448(%rsp)
        vmovups   %zmm25, 384(%rsp)
        vmovups   %zmm26, 320(%rsp)
        vmovups   %zmm27, 256(%rsp)
        vmovups   %zmm28, 192(%rsp)
        vmovups   %zmm29, 128(%rsp)
        vmovups   %zmm30, 64(%rsp)
        vmovups   %zmm31, (%rsp)
        movq      %rsi, 1056(%rsp)
        movq      %r12, 1096(%rsp)
        cfi_offset_rel_rsp (12, 1096)
        movb      %dl, %r12b
        movq      %r13, 1088(%rsp)
        cfi_offset_rel_rsp (13, 1088)
        movl      %eax, %r13d
        movq      %r14, 1080(%rsp)
        cfi_offset_rel_rsp (14, 1080)
        movl      %ecx, %r14d
        movq      %r15, 1072(%rsp)
        cfi_offset_rel_rsp (15, 1072)
        movq      %rbx, 1064(%rsp)
        movq      %rdi, %rbx
        cfi_remember_state

.LBL_2_6:
        btl       %r13d, %r14d
        jc        .LBL_2_13

.LBL_2_7:
        lea       1(%r13), %esi
        btl       %esi, %r14d
        jc        .LBL_2_10

.LBL_2_8:
        incb      %r12b
        addl      $2, %r13d
        cmpb      $16, %r12b
        jb        .LBL_2_6

        kmovw     1048(%rsp), %k4
        movq      %rbx, %rdi
        kmovw     1040(%rsp), %k5
        kmovw     1032(%rsp), %k6
        kmovw     1024(%rsp), %k7
        vmovups   960(%rsp), %zmm16
        vmovups   896(%rsp), %zmm17
        vmovups   832(%rsp), %zmm18
        vmovups   768(%rsp), %zmm19
        vmovups   704(%rsp), %zmm20
        vmovups   640(%rsp), %zmm21
        vmovups   576(%rsp), %zmm22
        vmovups   512(%rsp), %zmm23
        vmovups   448(%rsp), %zmm24
        vmovups   384(%rsp), %zmm25
        vmovups   320(%rsp), %zmm26
        vmovups   256(%rsp), %zmm27
        vmovups   192(%rsp), %zmm28
        vmovups   128(%rsp), %zmm29
        vmovups   64(%rsp), %zmm30
        vmovups   (%rsp), %zmm31
        vmovups   1216(%rsp), %zmm0
        vmovups   1280(%rsp), %zmm1
        movq      1056(%rsp), %rsi
        movq      1096(%rsp), %r12
        cfi_restore (%r12)
        movq      1088(%rsp), %r13
        cfi_restore (%r13)
        movq      1080(%rsp), %r14
        cfi_restore (%r14)
        movq      1072(%rsp), %r15
        cfi_restore (%r15)
        movq      1064(%rsp), %rbx
        jmp       .LBL_2_2

.LBL_2_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    1160(%rsp,%r15), %xmm0
        vzeroupper
        vmovsd    1160(%rsp,%r15), %xmm0

        call      JUMPTARGET(sin)

        vmovsd    %xmm0, 1224(%rsp,%r15)
        vmovsd    1160(%rsp,%r15), %xmm0

        call      JUMPTARGET(cos)

        vmovsd    %xmm0, 1288(%rsp,%r15)
        jmp       .LBL_2_8

.LBL_2_13:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    1152(%rsp,%r15), %xmm0
        vzeroupper
        vmovsd    1152(%rsp,%r15), %xmm0

        call      JUMPTARGET(sin)

        vmovsd    %xmm0, 1216(%rsp,%r15)
        vmovsd    1152(%rsp,%r15), %xmm0

        call      JUMPTARGET(cos)

        vmovsd    %xmm0, 1280(%rsp,%r15)
        jmp       .LBL_2_7

#endif
END (_ZGVeN8vl8l8_sincos_skx)
libmvec_hidden_def(_ZGVeN8vl8l8_sincos_skx)

/* Wrapper between vvv and vl8l8 vector variants.  */
.macro WRAPPER_AVX512_vvv_vl8l8 callee
#ifndef __ILP32__
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $256, %rsp
        vmovups   %zmm1, 128(%rsp)
        lea       (%rsp), %rdi
        vmovups   %zmm2, 192(%rdi)
        lea       64(%rsp), %rsi
        call      HIDDEN_JUMPTARGET(\callee)
        movq      128(%rsp), %rdx
        movq      136(%rsp), %rsi
        movq      144(%rsp), %r8
        movq      152(%rsp), %r10
        movq      (%rsp), %rax
        movq      8(%rsp), %rcx
        movq      16(%rsp), %rdi
        movq      24(%rsp), %r9
        movq      %rax, (%rdx)
        movq      %rcx, (%rsi)
        movq      160(%rsp), %rax
        movq      168(%rsp), %rcx
        movq      %rdi, (%r8)
        movq      %r9, (%r10)
        movq      176(%rsp), %rdi
        movq      184(%rsp), %r9
        movq      32(%rsp), %r11
        movq      40(%rsp), %rdx
        movq      48(%rsp), %rsi
        movq      56(%rsp), %r8
        movq      %r11, (%rax)
        movq      %rdx, (%rcx)
        movq      192(%rsp), %r11
        movq      200(%rsp), %rdx
        movq      %rsi, (%rdi)
        movq      %r8, (%r9)
        movq      208(%rsp), %rsi
        movq      216(%rsp), %r8
        movq      64(%rsp), %r10
        movq      72(%rsp), %rax
        movq      80(%rsp), %rcx
        movq      88(%rsp), %rdi
        movq      %r10, (%r11)
        movq      %rax, (%rdx)
        movq      224(%rsp), %r10
        movq      232(%rsp), %rax
        movq      %rcx, (%rsi)
        movq      %rdi, (%r8)
        movq      240(%rsp), %rcx
        movq      248(%rsp), %rdi
        movq      96(%rsp), %r9
        movq      104(%rsp), %r11
        movq      112(%rsp), %rdx
        movq      120(%rsp), %rsi
        movq      %r9, (%r10)
        movq      %r11, (%rax)
        movq      %rdx, (%rcx)
        movq      %rsi, (%rdi)
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret
#else
        leal    8(%rsp), %r10d
        .cfi_def_cfa 10, 0
        andl    $-64, %esp
        pushq   -8(%r10d)
        pushq   %rbp
        .cfi_escape 0x10,0x6,0x2,0x76,0
        movl    %esp, %ebp
        pushq   %r10
        .cfi_escape 0xf,0x3,0x76,0x78,0x6
        leal    -112(%rbp), %esi
        leal    -176(%rbp), %edi
        subl    $232, %esp
        vmovdqa %ymm1, -208(%ebp)
        vmovdqa %ymm2, -240(%ebp)
        call    HIDDEN_JUMPTARGET(\callee)
        vmovdqa -208(%ebp), %xmm0
        vmovq   %xmm0, %rax
        vmovsd  -176(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        shrq    $32, %rax
        vmovsd  -168(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movq    -200(%ebp), %rax
        vmovsd  -160(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        shrq    $32, %rax
        vmovsd  -152(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movq    -192(%ebp), %rax
        vmovsd  -144(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        shrq    $32, %rax
        vmovsd  -136(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movq    -184(%ebp), %rax
        vmovsd  -128(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        shrq    $32, %rax
        vmovsd  -120(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        vmovdqa -240(%ebp), %xmm0
        vmovq   %xmm0, %rax
        vmovsd  -112(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        shrq    $32, %rax
        vmovsd  -104(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movq    -232(%ebp), %rax
        vmovsd  -96(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        shrq    $32, %rax
        vmovsd  -88(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movq    -224(%ebp), %rax
        vmovsd  -80(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        shrq    $32, %rax
        vmovsd  -72(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movq    -216(%ebp), %rax
        vmovsd  -64(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        shrq    $32, %rax
        vmovsd  -56(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        addl    $232, %esp
        popq    %r10
        .cfi_def_cfa 10, 0
        popq    %rbp
        leal    -8(%r10), %esp
        .cfi_def_cfa 7, 8
        ret
#endif
.endm

ENTRY (_ZGVeN8vvv_sincos_knl)
WRAPPER_AVX512_vvv_vl8l8 _ZGVeN8vl8l8_sincos_knl
END (_ZGVeN8vvv_sincos_knl)

ENTRY (_ZGVeN8vvv_sincos_skx)
WRAPPER_AVX512_vvv_vl8l8 _ZGVeN8vl8l8_sincos_skx
END (_ZGVeN8vvv_sincos_skx)

	.section .rodata, "a"
.L_2il0floatpacket.15:
	.long	0xffffffff,0xffffffff
	.type	.L_2il0floatpacket.15,@object

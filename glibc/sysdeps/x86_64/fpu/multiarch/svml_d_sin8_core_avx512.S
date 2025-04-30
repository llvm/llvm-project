/* Function sin vectorized with AVX-512, KNL and SKX versions.
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

	.text
ENTRY (_ZGVeN8v_sin_knl)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512 _ZGVdN4v_sin
#else
/*
   ALGORITHM DESCRIPTION:

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
        subq      $1280, %rsp
        movq      __svml_d_trig_data@GOTPCREL(%rip), %rax
        movq      $-1, %rdx
        vmovups __dAbsMask(%rax), %zmm6
        vmovups __dInvPI(%rax), %zmm1

/*
   ARGUMENT RANGE REDUCTION:
   X' = |X|
 */
        vpandq    %zmm6, %zmm0, %zmm12
        vmovups __dPI1_FMA(%rax), %zmm2
        vmovups __dC7_sin(%rax), %zmm7

/* SignX - sign bit of X */
        vpandnq   %zmm0, %zmm6, %zmm11

/* R = X' - N*Pi1 */
        vmovaps   %zmm12, %zmm3

/* Y = X'*InvPi + RS : right shifter add */
        vfmadd213pd __dRShifter(%rax), %zmm12, %zmm1
        vcmppd    $22, __dRangeVal(%rax), %zmm12, %k1
        vpbroadcastq %rdx, %zmm13{%k1}{z}

/* N = Y - RS : right shifter sub */
        vsubpd __dRShifter(%rax), %zmm1, %zmm4

/* SignRes = Y<<63 : shift LSB to MSB place for result sign */
        vpsllq    $63, %zmm1, %zmm5
        vptestmq  %zmm13, %zmm13, %k0
        vfnmadd231pd %zmm4, %zmm2, %zmm3
        kmovw     %k0, %ecx
        movzbl    %cl, %ecx

/* R = R - N*Pi2 */
        vfnmadd231pd __dPI2_FMA(%rax), %zmm4, %zmm3

/* R = R - N*Pi3 */
        vfnmadd132pd __dPI3_FMA(%rax), %zmm3, %zmm4

/*
  POLYNOMIAL APPROXIMATION:
  R2 = R*R
 */
        vmulpd    %zmm4, %zmm4, %zmm8

/* R = R^SignRes : update sign of reduced argument */
        vpxorq    %zmm5, %zmm4, %zmm9
        vfmadd213pd __dC6_sin(%rax), %zmm8, %zmm7
        vfmadd213pd __dC5_sin(%rax), %zmm8, %zmm7
        vfmadd213pd __dC4_sin(%rax), %zmm8, %zmm7

/* Poly = C3+R2*(C4+R2*(C5+R2*(C6+R2*C7))) */
        vfmadd213pd __dC3_sin(%rax), %zmm8, %zmm7

/* Poly = R2*(C1+R2*(C2+R2*Poly)) */
        vfmadd213pd __dC2_sin(%rax), %zmm8, %zmm7
        vfmadd213pd __dC1_sin(%rax), %zmm8, %zmm7
        vmulpd    %zmm8, %zmm7, %zmm10

/* Poly = Poly*R + R */
        vfmadd213pd %zmm9, %zmm9, %zmm10

/*
   RECONSTRUCTION:
   Final sign setting: Res = Poly^SignX
 */
        vpxorq    %zmm11, %zmm10, %zmm1
        testl     %ecx, %ecx
        jne       .LBL_1_3

.LBL_1_2:
        cfi_remember_state
        vmovaps   %zmm1, %zmm0
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_1_3:
        cfi_restore_state
        vmovups   %zmm0, 1152(%rsp)
        vmovups   %zmm1, 1216(%rsp)
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
        movq      %rsi, 1064(%rsp)
        movq      %rdi, 1056(%rsp)
        movq      %r12, 1096(%rsp)
        cfi_offset_rel_rsp (12, 1096)
        movb      %dl, %r12b
        movq      %r13, 1088(%rsp)
        cfi_offset_rel_rsp (13, 1088)
        movl      %ecx, %r13d
        movq      %r14, 1080(%rsp)
        cfi_offset_rel_rsp (14, 1080)
        movl      %eax, %r14d
        movq      %r15, 1072(%rsp)
        cfi_offset_rel_rsp (15, 1072)
        cfi_remember_state

.LBL_1_6:
        btl       %r14d, %r13d
        jc        .LBL_1_12

.LBL_1_7:
        lea       1(%r14), %esi
        btl       %esi, %r13d
        jc        .LBL_1_10

.LBL_1_8:
        addb      $1, %r12b
        addl      $2, %r14d
        cmpb      $16, %r12b
        jb        .LBL_1_6

        kmovw     1048(%rsp), %k4
        movq      1064(%rsp), %rsi
        kmovw     1040(%rsp), %k5
        movq      1056(%rsp), %rdi
        kmovw     1032(%rsp), %k6
        movq      1096(%rsp), %r12
        cfi_restore (%r12)
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
        vmovups   1216(%rsp), %zmm1
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    1160(%rsp,%r15), %xmm0
        call      JUMPTARGET(sin)
        vmovsd    %xmm0, 1224(%rsp,%r15)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    1152(%rsp,%r15), %xmm0
        call      JUMPTARGET(sin)
        vmovsd    %xmm0, 1216(%rsp,%r15)
        jmp       .LBL_1_7
#endif
END (_ZGVeN8v_sin_knl)

ENTRY (_ZGVeN8v_sin_skx)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512 _ZGVdN4v_sin
#else
/*
   ALGORITHM DESCRIPTION:

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
        subq      $1280, %rsp
        movq      __svml_d_trig_data@GOTPCREL(%rip), %rax
        vpbroadcastq .L_2il0floatpacket.14(%rip), %zmm14
        vmovups __dAbsMask(%rax), %zmm7
        vmovups __dInvPI(%rax), %zmm2
        vmovups __dRShifter(%rax), %zmm1
        vmovups __dPI1_FMA(%rax), %zmm3
        vmovups __dC7_sin(%rax), %zmm8

/*
  ARGUMENT RANGE REDUCTION:
  X' = |X|
 */
        vandpd    %zmm7, %zmm0, %zmm13

/* SignX - sign bit of X */
        vandnpd   %zmm0, %zmm7, %zmm12

/* Y = X'*InvPi + RS : right shifter add */
        vfmadd213pd %zmm1, %zmm13, %zmm2
        vcmppd    $18, __dRangeVal(%rax), %zmm13, %k1

/* SignRes = Y<<63 : shift LSB to MSB place for result sign */
        vpsllq    $63, %zmm2, %zmm6

/* N = Y - RS : right shifter sub */
        vsubpd    %zmm1, %zmm2, %zmm5

/* R = X' - N*Pi1 */
        vmovaps   %zmm13, %zmm4
        vfnmadd231pd %zmm5, %zmm3, %zmm4

/* R = R - N*Pi2 */
        vfnmadd231pd __dPI2_FMA(%rax), %zmm5, %zmm4

/* R = R - N*Pi3 */
        vfnmadd132pd __dPI3_FMA(%rax), %zmm4, %zmm5

/*
  POLYNOMIAL APPROXIMATION:
  R2 = R*R
 */
        vmulpd    %zmm5, %zmm5, %zmm9

/* R = R^SignRes : update sign of reduced argument */
        vxorpd    %zmm6, %zmm5, %zmm10
        vfmadd213pd __dC6_sin(%rax), %zmm9, %zmm8
        vfmadd213pd __dC5_sin(%rax), %zmm9, %zmm8
        vfmadd213pd __dC4_sin(%rax), %zmm9, %zmm8

/* Poly = C3+R2*(C4+R2*(C5+R2*(C6+R2*C7))) */
        vfmadd213pd __dC3_sin(%rax), %zmm9, %zmm8

/* Poly = R2*(C1+R2*(C2+R2*Poly)) */
        vfmadd213pd __dC2_sin(%rax), %zmm9, %zmm8
        vfmadd213pd __dC1_sin(%rax), %zmm9, %zmm8
        vmulpd    %zmm9, %zmm8, %zmm11

/* Poly = Poly*R + R */
        vfmadd213pd %zmm10, %zmm10, %zmm11

/*
  RECONSTRUCTION:
  Final sign setting: Res = Poly^SignX
 */
        vxorpd    %zmm12, %zmm11, %zmm1
        vpandnq   %zmm13, %zmm13, %zmm14{%k1}
        vcmppd    $3, %zmm14, %zmm14, %k0
        kmovw     %k0, %ecx
        testl     %ecx, %ecx
        jne       .LBL_2_3

.LBL_2_2:
        cfi_remember_state
        vmovaps   %zmm1, %zmm0
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_2_3:
        cfi_restore_state
        vmovups   %zmm0, 1152(%rsp)
        vmovups   %zmm1, 1216(%rsp)
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
        movq      %rsi, 1064(%rsp)
        movq      %rdi, 1056(%rsp)
        movq      %r12, 1096(%rsp)
        cfi_offset_rel_rsp (12, 1096)
        movb      %dl, %r12b
        movq      %r13, 1088(%rsp)
        cfi_offset_rel_rsp (13, 1088)
        movl      %ecx, %r13d
        movq      %r14, 1080(%rsp)
        cfi_offset_rel_rsp (14, 1080)
        movl      %eax, %r14d
        movq      %r15, 1072(%rsp)
        cfi_offset_rel_rsp (15, 1072)
        cfi_remember_state

.LBL_2_6:
        btl       %r14d, %r13d
        jc        .LBL_2_12

.LBL_2_7:
        lea       1(%r14), %esi
        btl       %esi, %r13d
        jc        .LBL_2_10

.LBL_2_8:
        incb      %r12b
        addl      $2, %r14d
        cmpb      $16, %r12b
        jb        .LBL_2_6

        kmovw     1048(%rsp), %k4
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
        vmovups   1216(%rsp), %zmm1
        movq      1064(%rsp), %rsi
        movq      1056(%rsp), %rdi
        movq      1096(%rsp), %r12
        cfi_restore (%r12)
        movq      1088(%rsp), %r13
        cfi_restore (%r13)
        movq      1080(%rsp), %r14
        cfi_restore (%r14)
        movq      1072(%rsp), %r15
        cfi_restore (%r15)
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
        jmp       .LBL_2_8

.LBL_2_12:
        movzbl    %r12b, %r15d
        shlq      $4, %r15
        vmovsd    1152(%rsp,%r15), %xmm0
        vzeroupper
        vmovsd    1152(%rsp,%r15), %xmm0

        call      JUMPTARGET(sin)

        vmovsd    %xmm0, 1216(%rsp,%r15)
        jmp       .LBL_2_7
#endif
END (_ZGVeN8v_sin_skx)

	.section .rodata, "a"
.L_2il0floatpacket.14:
	.long	0xffffffff,0xffffffff
	.type	.L_2il0floatpacket.14,@object

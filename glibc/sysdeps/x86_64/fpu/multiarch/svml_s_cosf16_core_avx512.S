/* Function cosf vectorized with AVX-512. KNL and SKX versions.
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
#include "svml_s_wrapper_impl.h"

	.text
ENTRY (_ZGVeN16v_cosf_knl)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512 _ZGVdN8v_cosf
#else
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
        subq      $1280, %rsp
        movq      __svml_s_trig_data@GOTPCREL(%rip), %rdx

/*
  h) Subtract Y*PI from X argument, where PI divided to 4 parts:
  X = X - Y*PI1 - Y*PI2 - Y*PI3
 */
        vmovaps   %zmm0, %zmm6
        movl      $-1, %eax

/* b) Add Pi/2 value to argument X for Cos to Sin transformation */
        vaddps    __sHalfPI(%rdx), %zmm0, %zmm2
        vmovups   __sRShifter(%rdx), %zmm3

/*
  1) Range reduction to [-Pi/2; +Pi/2] interval
  c) Getting octant Y by 1/Pi multiplication
  d) Add "Right Shifter" (0x4B000000) value
 */
        vfmadd132ps __sInvPI(%rdx), %zmm3, %zmm2
        vmovups     __sPI1_FMA(%rdx), %zmm5

/* f) Subtract "Right Shifter" (0x4B000000) value */
        vsubps    %zmm3, %zmm2, %zmm4
        vmovups   __sA9_FMA(%rdx), %zmm9

/* Check for large and special arguments */
        vpandd    __sAbsMask(%rdx), %zmm0, %zmm1

/*
  e) Treat obtained value as integer for destination sign setting.
  Shift first bit of this value to the last (sign) position (S << 31)
 */
        vpslld       $31, %zmm2, %zmm8
        vcmpps       $22, __sRangeReductionVal(%rdx), %zmm1, %k1
        vpbroadcastd %eax, %zmm12{%k1}{z}

/* g) Subtract 0.5 from result for octant correction */
        vsubps       __sOneHalf(%rdx), %zmm4, %zmm7
        vptestmd     %zmm12, %zmm12, %k0
        vfnmadd231ps %zmm7, %zmm5, %zmm6
        kmovw        %k0, %ecx
        vfnmadd231ps __sPI2_FMA(%rdx), %zmm7, %zmm6
        vfnmadd132ps __sPI3_FMA(%rdx), %zmm6, %zmm7

/* a) Calculate X^2 = X * X */
        vmulps    %zmm7, %zmm7, %zmm10

/*
  3) Destination sign setting
    a) Set shifted destination sign using XOR operation:
  R = XOR( R, S );
 */
        vpxord    %zmm8, %zmm7, %zmm11

/*
  b) Calculate polynomial:
  R = X + X * X^2 * (A3 + x^2 * (A5 + x^2 * (A7 + x^2 * (A9))));
 */
        vfmadd213ps __sA7_FMA(%rdx), %zmm10, %zmm9
        vfmadd213ps __sA5_FMA(%rdx), %zmm10, %zmm9
        vfmadd213ps __sA3(%rdx), %zmm10, %zmm9
        vmulps      %zmm10, %zmm9, %zmm1
        vfmadd213ps %zmm11, %zmm11, %zmm1
        testl       %ecx, %ecx
        jne         .LBL_1_3

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
        vmovss    1156(%rsp,%r15,8), %xmm0
        call      JUMPTARGET(cosf)
        vmovss    %xmm0, 1220(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        vmovss    1152(%rsp,%r15,8), %xmm0
        call      JUMPTARGET(cosf)
        vmovss    %xmm0, 1216(%rsp,%r15,8)
        jmp       .LBL_1_7
#endif
END (_ZGVeN16v_cosf_knl)

ENTRY (_ZGVeN16v_cosf_skx)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512 _ZGVdN8v_cosf
#else
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
        subq      $1280, %rsp
        movq      __svml_s_trig_data@GOTPCREL(%rip), %rax

/*
  h) Subtract Y*PI from X argument, where PI divided to 4 parts:
  X = X - Y*PI1 - Y*PI2 - Y*PI3
 */
        vmovaps   %zmm0, %zmm6
        vmovups   .L_2il0floatpacket.13(%rip), %zmm12
        vmovups __sRShifter(%rax), %zmm3
        vmovups __sPI1_FMA(%rax), %zmm5
        vmovups __sA9_FMA(%rax), %zmm9

/* b) Add Pi/2 value to argument X for Cos to Sin transformation */
        vaddps __sHalfPI(%rax), %zmm0, %zmm2

/* Check for large and special arguments */
        vandps __sAbsMask(%rax), %zmm0, %zmm1

/*
  1) Range reduction to [-Pi/2; +Pi/2] interval
  c) Getting octant Y by 1/Pi multiplication
  d) Add "Right Shifter" (0x4B000000) value
 */
        vfmadd132ps __sInvPI(%rax), %zmm3, %zmm2
        vcmpps    $18, __sRangeReductionVal(%rax), %zmm1, %k1

/*
  e) Treat obtained value as integer for destination sign setting.
  Shift first bit of this value to the last (sign) position (S << 31)
 */
        vpslld    $31, %zmm2, %zmm8

/* f) Subtract "Right Shifter" (0x4B000000) value */
        vsubps    %zmm3, %zmm2, %zmm4

/* g) Subtract 0.5 from result for octant correction */
        vsubps __sOneHalf(%rax), %zmm4, %zmm7
        vfnmadd231ps %zmm7, %zmm5, %zmm6
        vfnmadd231ps __sPI2_FMA(%rax), %zmm7, %zmm6
        vfnmadd132ps __sPI3_FMA(%rax), %zmm6, %zmm7

/* a) Calculate X^2 = X * X */
        vmulps    %zmm7, %zmm7, %zmm10

/*
  3) Destination sign setting
  a) Set shifted destination sign using XOR operation:
  R = XOR( R, S );
 */
        vxorps    %zmm8, %zmm7, %zmm11

/*
  b) Calculate polynomial:
  R = X + X * X^2 * (A3 + x^2 * (A5 + x^2 * (A7 + x^2 * (A9))));
 */
        vfmadd213ps __sA7_FMA(%rax), %zmm10, %zmm9
        vfmadd213ps __sA5_FMA(%rax), %zmm10, %zmm9
        vfmadd213ps __sA3(%rax), %zmm10, %zmm9
        vpandnd   %zmm1, %zmm1, %zmm12{%k1}
        vmulps    %zmm10, %zmm9, %zmm1
        vptestmd  %zmm12, %zmm12, %k0
        vfmadd213ps %zmm11, %zmm11, %zmm1
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
        vmovss    1156(%rsp,%r15,8), %xmm0
        vzeroupper
        vmovss    1156(%rsp,%r15,8), %xmm0
        call      JUMPTARGET(cosf)
        vmovss    %xmm0, 1220(%rsp,%r15,8)
        jmp       .LBL_2_8
.LBL_2_12:
        movzbl    %r12b, %r15d
        vmovss    1152(%rsp,%r15,8), %xmm0
        vzeroupper
        vmovss    1152(%rsp,%r15,8), %xmm0
        call      JUMPTARGET(cosf)
        vmovss    %xmm0, 1216(%rsp,%r15,8)
        jmp       .LBL_2_7
#endif
END (_ZGVeN16v_cosf_skx)

	.section .rodata, "a"
.L_2il0floatpacket.13:
	.long	0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff
	.type	.L_2il0floatpacket.13,@object

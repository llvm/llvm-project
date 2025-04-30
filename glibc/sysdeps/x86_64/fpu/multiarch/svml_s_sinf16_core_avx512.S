/* Function sinf vectorized with AVX-512. KNL and SKX versions.
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
ENTRY(_ZGVeN16v_sinf_knl)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512 _ZGVdN8v_sinf
#else
/*
   ALGORITHM DESCRIPTION:

   1) Range reduction to [-Pi/2; +Pi/2] interval
      a) Grab sign from source argument and save it.
      b) Remove sign using AND operation
      c) Getting octant Y by 1/Pi multiplication
      d) Add "Right Shifter" value
      e) Treat obtained value as integer for destination sign setting.
         Shift first bit of this value to the last (sign) position
      f) Change destination sign if source sign is negative
         using XOR operation.
      g) Subtract "Right Shifter" value
      h) Subtract Y*PI from X argument, where PI divided to 4 parts:
         X = X - Y*PI1 - Y*PI2 - Y*PI3 - Y*PI4;
   2) Polynomial (minimax for sin within [-Pi/2; +Pi/2] interval)
      a) Calculate X^2 = X * X
      b) Calculate polynomial:
         R = X + X * X^2 * (A3 + x^2 * (A5 + ......
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

/* Check for large and special values */
        movl      $-1, %edx
        vmovups __sAbsMask(%rax), %zmm4
        vmovups __sInvPI(%rax), %zmm1

/* b) Remove sign using AND operation */
        vpandd    %zmm4, %zmm0, %zmm12
        vmovups __sPI1_FMA(%rax), %zmm2
        vmovups __sA9(%rax), %zmm7

/*
  f) Change destination sign if source sign is negative
  using XOR operation.
 */
        vpandnd   %zmm0, %zmm4, %zmm11

/*
  h) Subtract Y*PI from X argument, where PI divided to 4 parts:
  X = X - Y*PI1 - Y*PI2 - Y*PI3;
 */
        vmovaps   %zmm12, %zmm3

/*
  c) Getting octant Y by 1/Pi multiplication
  d) Add "Right Shifter" value
 */
        vfmadd213ps __sRShifter(%rax), %zmm12, %zmm1
        vcmpps    $22, __sRangeReductionVal(%rax), %zmm12, %k1
        vpbroadcastd %edx, %zmm13{%k1}{z}

/* g) Subtract "Right Shifter" value */
        vsubps __sRShifter(%rax), %zmm1, %zmm5

/*
  e) Treat obtained value as integer for destination sign setting.
  Shift first bit of this value to the last (sign) position
 */
        vpslld    $31, %zmm1, %zmm6
        vptestmd  %zmm13, %zmm13, %k0
        vfnmadd231ps %zmm5, %zmm2, %zmm3
        kmovw     %k0, %ecx
        vfnmadd231ps __sPI2_FMA(%rax), %zmm5, %zmm3
        vfnmadd132ps __sPI3_FMA(%rax), %zmm3, %zmm5

/*
  2) Polynomial (minimax for sin within [-Pi/2; +Pi/2] interval)
  a) Calculate X^2 = X * X
  b) Calculate polynomial:
  R = X + X * X^2 * (A3 + x^2 * (A5 + ......
 */
        vmulps    %zmm5, %zmm5, %zmm8
        vpxord    %zmm6, %zmm5, %zmm9
        vfmadd213ps __sA7(%rax), %zmm8, %zmm7
        vfmadd213ps __sA5(%rax), %zmm8, %zmm7
        vfmadd213ps __sA3(%rax), %zmm8, %zmm7
        vmulps    %zmm8, %zmm7, %zmm10
        vfmadd213ps %zmm9, %zmm9, %zmm10

/*
  3) Destination sign setting
  a) Set shifted destination sign using XOR operation:
  R = XOR( R, S );
 */
        vpxord    %zmm11, %zmm10, %zmm1
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
        vmovss    1156(%rsp,%r15,8), %xmm0
        call      JUMPTARGET(sinf)
        vmovss    %xmm0, 1220(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_12:
        movzbl    %r12b, %r15d
        vmovss    1152(%rsp,%r15,8), %xmm0
        call      JUMPTARGET(sinf)
        vmovss    %xmm0, 1216(%rsp,%r15,8)
        jmp       .LBL_1_7
#endif
END(_ZGVeN16v_sinf_knl)

ENTRY (_ZGVeN16v_sinf_skx)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512 _ZGVdN8v_sinf
#else
/*
   ALGORITHM DESCRIPTION:

   1) Range reduction to [-Pi/2; +Pi/2] interval
      a) Grab sign from source argument and save it.
      b) Remove sign using AND operation
      c) Getting octant Y by 1/Pi multiplication
      d) Add "Right Shifter" value
      e) Treat obtained value as integer for destination sign setting.
         Shift first bit of this value to the last (sign) position
      f) Change destination sign if source sign is negative
         using XOR operation.
      g) Subtract "Right Shifter" value
      h) Subtract Y*PI from X argument, where PI divided to 4 parts:
         X = X - Y*PI1 - Y*PI2 - Y*PI3 - Y*PI4;
   2) Polynomial (minimax for sin within [-Pi/2; +Pi/2] interval)
      a) Calculate X^2 = X * X
      b) Calculate polynomial:
         R = X + X * X^2 * (A3 + x^2 * (A5 + ......
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

/* Check for large and special values */
        vmovups   .L_2il0floatpacket.11(%rip), %zmm14
        vmovups __sAbsMask(%rax), %zmm5
        vmovups __sInvPI(%rax), %zmm1
        vmovups __sRShifter(%rax), %zmm2
        vmovups __sPI1_FMA(%rax), %zmm3
        vmovups __sA9(%rax), %zmm8

/* b) Remove sign using AND operation */
        vandps    %zmm5, %zmm0, %zmm13

/*
  f) Change destination sign if source sign is negative
  using XOR operation.
 */
        vandnps   %zmm0, %zmm5, %zmm12

/*
  c) Getting octant Y by 1/Pi multiplication
  d) Add "Right Shifter" value
 */
        vfmadd213ps %zmm2, %zmm13, %zmm1
        vcmpps    $18, __sRangeReductionVal(%rax), %zmm13, %k1

/*
  e) Treat obtained value as integer for destination sign setting.
  Shift first bit of this value to the last (sign) position
 */
        vpslld    $31, %zmm1, %zmm7

/* g) Subtract "Right Shifter" value */
        vsubps    %zmm2, %zmm1, %zmm6

/*
  h) Subtract Y*PI from X argument, where PI divided to 4 parts:
  X = X - Y*PI1 - Y*PI2 - Y*PI3;
 */
        vmovaps   %zmm13, %zmm4
        vfnmadd231ps %zmm6, %zmm3, %zmm4
        vfnmadd231ps __sPI2_FMA(%rax), %zmm6, %zmm4
        vfnmadd132ps __sPI3_FMA(%rax), %zmm4, %zmm6

/*
  2) Polynomial (minimax for sin within [-Pi/2; +Pi/2] interval)
  a) Calculate X^2 = X * X
  b) Calculate polynomial:
  R = X + X * X^2 * (A3 + x^2 * (A5 + ......
 */
        vmulps    %zmm6, %zmm6, %zmm9
        vxorps    %zmm7, %zmm6, %zmm10
        vfmadd213ps __sA7(%rax), %zmm9, %zmm8
        vfmadd213ps __sA5(%rax), %zmm9, %zmm8
        vfmadd213ps __sA3(%rax), %zmm9, %zmm8
        vmulps    %zmm9, %zmm8, %zmm11
        vfmadd213ps %zmm10, %zmm10, %zmm11

/*
  3) Destination sign setting
  a) Set shifted destination sign using XOR operation:
  R = XOR( R, S );
 */
        vxorps    %zmm12, %zmm11, %zmm1
        vpandnd   %zmm13, %zmm13, %zmm14{%k1}
        vptestmd  %zmm14, %zmm14, %k0
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

        call      JUMPTARGET(sinf)

        vmovss    %xmm0, 1220(%rsp,%r15,8)
        jmp       .LBL_2_8

.LBL_2_12:
        movzbl    %r12b, %r15d
        vmovss    1152(%rsp,%r15,8), %xmm0
        vzeroupper
        vmovss    1152(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(sinf)

        vmovss    %xmm0, 1216(%rsp,%r15,8)
        jmp       .LBL_2_7
#endif
END (_ZGVeN16v_sinf_skx)

	.section .rodata, "a"
.L_2il0floatpacket.11:
	.long	0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff
	.type	.L_2il0floatpacket.11,@object

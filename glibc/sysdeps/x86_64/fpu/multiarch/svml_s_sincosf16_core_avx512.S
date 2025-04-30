/* Function sincosf vectorized with AVX-512. KNL and SKX versions.
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

	.text
ENTRY (_ZGVeN16vl4l4_sincosf_knl)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512_fFF _ZGVdN8vl4l4_sincosf
#else
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $1344, %rsp
        movq      __svml_s_trig_data@GOTPCREL(%rip), %rax
        vmovaps   %zmm0, %zmm2
        movl      $-1, %edx
        vmovups __sAbsMask(%rax), %zmm0
        vmovups __sInvPI(%rax), %zmm3

/* Absolute argument computation */
        vpandd    %zmm0, %zmm2, %zmm1
        vmovups __sPI1_FMA(%rax), %zmm5
        vmovups __sSignMask(%rax), %zmm9
        vpandnd   %zmm2, %zmm0, %zmm0

/* h) Subtract Y*(PI/2) from X argument, where PI/2 divided to 3 parts:
      X = X - Y*PI1 - Y*PI2 - Y*PI3 */
        vmovaps   %zmm1, %zmm6
        vmovaps   %zmm1, %zmm8

/* c) Getting octant Y by 2/Pi multiplication
   d) Add "Right Shifter" value */
        vfmadd213ps __sRShifter(%rax), %zmm1, %zmm3
        vmovups __sPI3_FMA(%rax), %zmm7

/* g) Subtract "Right Shifter" (0x4B000000) value */
        vsubps __sRShifter(%rax), %zmm3, %zmm12

/* e) Treat obtained value as integer S for destination sign setting */
        vpslld    $31, %zmm3, %zmm13
        vmovups __sA7_FMA(%rax), %zmm14
        vfnmadd231ps %zmm12, %zmm5, %zmm6

/* 2) Polynomial (minimax for sin within  [-Pi/4; +Pi/4] interval)
      a) Calculate X^2 = X * X
      b) Calculate 2 polynomials for sin and cos:
         RS = X * ( A0 + X^2 * (A1 + x^2 * (A2 + x^2 * (A3))));
         RC = B0 + X^2 * (B1 + x^2 * (B2 + x^2 * (B3 + x^2 * (B4)))) */
        vmovaps   %zmm14, %zmm15
        vmovups __sA9_FMA(%rax), %zmm3
        vcmpps    $22, __sRangeReductionVal(%rax), %zmm1, %k1
        vpbroadcastd %edx, %zmm1{%k1}{z}
        vfnmadd231ps __sPI2_FMA(%rax), %zmm12, %zmm6
        vptestmd  %zmm1, %zmm1, %k0
        vpandd    %zmm6, %zmm9, %zmm11
        kmovw     %k0, %ecx
        vpxord __sOneHalf(%rax), %zmm11, %zmm4

/* Result sign calculations */
        vpternlogd $150, %zmm13, %zmm9, %zmm11

/* Add correction term 0.5 for cos() part */
        vaddps    %zmm4, %zmm12, %zmm10
        vfnmadd213ps %zmm6, %zmm7, %zmm12
        vfnmadd231ps %zmm10, %zmm5, %zmm8
        vpxord    %zmm13, %zmm12, %zmm13
        vmulps    %zmm13, %zmm13, %zmm12
        vfnmadd231ps __sPI2_FMA(%rax), %zmm10, %zmm8
        vfmadd231ps __sA9_FMA(%rax), %zmm12, %zmm15
        vfnmadd213ps %zmm8, %zmm7, %zmm10
        vfmadd213ps __sA5_FMA(%rax), %zmm12, %zmm15
        vpxord    %zmm11, %zmm10, %zmm5
        vmulps    %zmm5, %zmm5, %zmm4
        vfmadd213ps __sA3(%rax), %zmm12, %zmm15
        vfmadd213ps %zmm14, %zmm4, %zmm3
        vmulps    %zmm12, %zmm15, %zmm14
        vfmadd213ps __sA5_FMA(%rax), %zmm4, %zmm3
        vfmadd213ps %zmm13, %zmm13, %zmm14
        vfmadd213ps __sA3(%rax), %zmm4, %zmm3
        vpxord    %zmm0, %zmm14, %zmm0
        vmulps    %zmm4, %zmm3, %zmm3
        vfmadd213ps %zmm5, %zmm5, %zmm3
        testl     %ecx, %ecx
        jne       .LBL_1_3

.LBL_1_2:
        cfi_remember_state
        vmovups   %zmm0, (%rdi)
        vmovups   %zmm3, (%rsi)
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_1_3:
        cfi_restore_state
        vmovups   %zmm2, 1152(%rsp)
        vmovups   %zmm0, 1216(%rsp)
        vmovups   %zmm3, 1280(%rsp)
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
        vmovups   1280(%rsp), %zmm3
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        vmovss    1156(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(sinf)

        vmovss    %xmm0, 1220(%rsp,%r15,8)
        vmovss    1156(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(cosf)

        vmovss    %xmm0, 1284(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_13:
        movzbl    %r12b, %r15d
        vmovss    1152(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(sinf)

        vmovss    %xmm0, 1216(%rsp,%r15,8)
        vmovss    1152(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(cosf)

        vmovss    %xmm0, 1280(%rsp,%r15,8)
        jmp       .LBL_1_7
#endif
END (_ZGVeN16vl4l4_sincosf_knl)
libmvec_hidden_def(_ZGVeN16vl4l4_sincosf_knl)

ENTRY (_ZGVeN16vl4l4_sincosf_skx)
#ifndef HAVE_AVX512DQ_ASM_SUPPORT
WRAPPER_IMPL_AVX512_fFF _ZGVdN8vvv_sincosf
#else
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $1344, %rsp
        movq      __svml_s_trig_data@GOTPCREL(%rip), %rax
        vmovaps   %zmm0, %zmm4
        vmovups __sAbsMask(%rax), %zmm3
        vmovups __sInvPI(%rax), %zmm5
        vmovups __sRShifter(%rax), %zmm6
        vmovups __sPI1_FMA(%rax), %zmm9
        vmovups __sPI2_FMA(%rax), %zmm10
        vmovups __sSignMask(%rax), %zmm14
        vmovups __sOneHalf(%rax), %zmm7
        vmovups __sPI3_FMA(%rax), %zmm12

/* Absolute argument computation */
        vandps    %zmm3, %zmm4, %zmm2

/* c) Getting octant Y by 2/Pi multiplication
   d) Add "Right Shifter" value */
        vfmadd213ps %zmm6, %zmm2, %zmm5
        vcmpps    $18, __sRangeReductionVal(%rax), %zmm2, %k1

/* e) Treat obtained value as integer S for destination sign setting */
        vpslld    $31, %zmm5, %zmm0

/* g) Subtract "Right Shifter" (0x4B000000) value */
        vsubps    %zmm6, %zmm5, %zmm5
        vmovups __sA3(%rax), %zmm6

/* h) Subtract Y*(PI/2) from X argument, where PI/2 divided to 3 parts:
      X = X - Y*PI1 - Y*PI2 - Y*PI3 */
        vmovaps   %zmm2, %zmm11
        vfnmadd231ps %zmm5, %zmm9, %zmm11
        vfnmadd231ps %zmm5, %zmm10, %zmm11
        vandps    %zmm11, %zmm14, %zmm1
        vxorps    %zmm1, %zmm7, %zmm8

/* Result sign calculations */
        vpternlogd $150, %zmm0, %zmm14, %zmm1
        vmovups   .L_2il0floatpacket.13(%rip), %zmm14

/* Add correction term 0.5 for cos() part */
        vaddps    %zmm8, %zmm5, %zmm15
        vfnmadd213ps %zmm11, %zmm12, %zmm5
        vandnps   %zmm4, %zmm3, %zmm11
        vmovups __sA7_FMA(%rax), %zmm3
        vmovaps   %zmm2, %zmm13
        vfnmadd231ps %zmm15, %zmm9, %zmm13
        vxorps    %zmm0, %zmm5, %zmm9
        vmovups __sA5_FMA(%rax), %zmm0
        vfnmadd231ps %zmm15, %zmm10, %zmm13
        vmulps    %zmm9, %zmm9, %zmm8
        vfnmadd213ps %zmm13, %zmm12, %zmm15
        vmovups __sA9_FMA(%rax), %zmm12
        vxorps    %zmm1, %zmm15, %zmm1
        vmulps    %zmm1, %zmm1, %zmm13

/* 2) Polynomial (minimax for sin within  [-Pi/4; +Pi/4] interval)
      a) Calculate X^2 = X * X
      b) Calculate 2 polynomials for sin and cos:
         RS = X * ( A0 + X^2 * (A1 + x^2 * (A2 + x^2 * (A3))));
         RC = B0 + X^2 * (B1 + x^2 * (B2 + x^2 * (B3 + x^2 * (B4)))) */
        vmovaps   %zmm12, %zmm7
        vfmadd213ps %zmm3, %zmm8, %zmm7
        vfmadd213ps %zmm3, %zmm13, %zmm12
        vfmadd213ps %zmm0, %zmm8, %zmm7
        vfmadd213ps %zmm0, %zmm13, %zmm12
        vfmadd213ps %zmm6, %zmm8, %zmm7
        vfmadd213ps %zmm6, %zmm13, %zmm12
        vmulps    %zmm8, %zmm7, %zmm10
        vmulps    %zmm13, %zmm12, %zmm3
        vfmadd213ps %zmm9, %zmm9, %zmm10
        vfmadd213ps %zmm1, %zmm1, %zmm3
        vxorps    %zmm11, %zmm10, %zmm0
        vpandnd   %zmm2, %zmm2, %zmm14{%k1}
        vptestmd  %zmm14, %zmm14, %k0
        kmovw     %k0, %ecx
        testl     %ecx, %ecx
        jne       .LBL_2_3

.LBL_2_2:
        cfi_remember_state
        vmovups   %zmm0, (%rdi)
        vmovups   %zmm3, (%rsi)
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_2_3:
        cfi_restore_state
        vmovups   %zmm4, 1152(%rsp)
        vmovups   %zmm0, 1216(%rsp)
        vmovups   %zmm3, 1280(%rsp)
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
        vmovups   1280(%rsp), %zmm3
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
        vmovss    1156(%rsp,%r15,8), %xmm0
        vzeroupper
        vmovss    1156(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(sinf)

        vmovss    %xmm0, 1220(%rsp,%r15,8)
        vmovss    1156(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(cosf)

        vmovss    %xmm0, 1284(%rsp,%r15,8)
        jmp       .LBL_2_8

.LBL_2_13:
        movzbl    %r12b, %r15d
        vmovss    1152(%rsp,%r15,8), %xmm0
        vzeroupper
        vmovss    1152(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(sinf)

        vmovss    %xmm0, 1216(%rsp,%r15,8)
        vmovss    1152(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(cosf)

        vmovss    %xmm0, 1280(%rsp,%r15,8)
        jmp       .LBL_2_7
#endif
END (_ZGVeN16vl4l4_sincosf_skx)
libmvec_hidden_def(_ZGVeN16vl4l4_sincosf_skx)

/* Wrapper between vvv and vl4l4 vector variants.  */
.macro WRAPPER_AVX512_vvv_vl4l4 callee
#ifndef __ILP32__
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $384, %rsp
        vmovups   %zmm1, 128(%rsp)
        lea       (%rsp), %rdi
        vmovups   %zmm2, 192(%rdi)
        vmovups   %zmm3, 256(%rdi)
        vmovups   %zmm4, 320(%rdi)
        lea       64(%rsp), %rsi
        call      HIDDEN_JUMPTARGET(\callee)
        movq      128(%rsp), %rdx
        movq      136(%rsp), %rsi
        movq      144(%rsp), %r8
        movq      152(%rsp), %r10
        movl      (%rsp), %eax
        movl      4(%rsp), %ecx
        movl      8(%rsp), %edi
        movl      12(%rsp), %r9d
        movl      %eax, (%rdx)
        movl      %ecx, (%rsi)
        movq      160(%rsp), %rax
        movq      168(%rsp), %rcx
        movl      %edi, (%r8)
        movl      %r9d, (%r10)
        movq      176(%rsp), %rdi
        movq      184(%rsp), %r9
        movl      16(%rsp), %r11d
        movl      20(%rsp), %edx
        movl      24(%rsp), %esi
        movl      28(%rsp), %r8d
        movl      %r11d, (%rax)
        movl      %edx, (%rcx)
        movq      192(%rsp), %r11
        movq      200(%rsp), %rdx
        movl      %esi, (%rdi)
        movl      %r8d, (%r9)
        movq      208(%rsp), %rsi
        movq      216(%rsp), %r8
        movl      32(%rsp), %r10d
        movl      36(%rsp), %eax
        movl      40(%rsp), %ecx
        movl      44(%rsp), %edi
        movl      %r10d, (%r11)
        movl      %eax, (%rdx)
        movq      224(%rsp), %r10
        movq      232(%rsp), %rax
        movl      %ecx, (%rsi)
        movl      %edi, (%r8)
        movq      240(%rsp), %rcx
        movq      248(%rsp), %rdi
        movl      48(%rsp), %r9d
        movl      52(%rsp), %r11d
        movl      56(%rsp), %edx
        movl      60(%rsp), %esi
        movl      %r9d, (%r10)
        movl      %r11d, (%rax)
        movq      256(%rsp), %r9
        movq      264(%rsp), %r11
        movl      %edx, (%rcx)
        movl      %esi, (%rdi)
        movq      272(%rsp), %rdx
        movq      280(%rsp), %rsi
        movl      64(%rsp), %r8d
        movl      68(%rsp), %r10d
        movl      72(%rsp), %eax
        movl      76(%rsp), %ecx
        movl      %r8d, (%r9)
        movl      %r10d, (%r11)
        movq      288(%rsp), %r8
        movq      296(%rsp), %r10
        movl      %eax, (%rdx)
        movl      %ecx, (%rsi)
        movq      304(%rsp), %rax
        movq      312(%rsp), %rcx
        movl      80(%rsp), %edi
        movl      84(%rsp), %r9d
        movl      88(%rsp), %r11d
        movl      92(%rsp), %edx
        movl      %edi, (%r8)
        movl      %r9d, (%r10)
        movq      320(%rsp), %rdi
        movq      328(%rsp), %r9
        movl      %r11d, (%rax)
        movl      %edx, (%rcx)
        movq      336(%rsp), %r11
        movq      344(%rsp), %rdx
        movl      96(%rsp), %esi
        movl      100(%rsp), %r8d
        movl      104(%rsp), %r10d
        movl      108(%rsp), %eax
        movl      %esi, (%rdi)
        movl      %r8d, (%r9)
        movq      352(%rsp), %rsi
        movq      360(%rsp), %r8
        movl      %r10d, (%r11)
        movl      %eax, (%rdx)
        movq      368(%rsp), %r10
        movq      376(%rsp), %rax
        movl      112(%rsp), %ecx
        movl      116(%rsp), %edi
        movl      120(%rsp), %r9d
        movl      124(%rsp), %r11d
        movl      %ecx, (%rsi)
        movl      %edi, (%r8)
        movl      %r9d, (%r10)
        movl      %r11d, (%rax)
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
        subl    $296, %esp
        vmovdqa64 %zmm1, -240(%ebp)
        vmovdqa64 %zmm2, -304(%ebp)
        call    HIDDEN_JUMPTARGET(\callee)
        movl    -240(%ebp), %eax
        vmovss  -176(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -236(%ebp), %eax
        vmovss  -172(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -232(%ebp), %eax
        vmovss  -168(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -228(%ebp), %eax
        vmovss  -164(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -224(%ebp), %eax
        vmovss  -160(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -220(%ebp), %eax
        vmovss  -156(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -216(%ebp), %eax
        vmovss  -152(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -212(%ebp), %eax
        vmovss  -148(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -208(%ebp), %eax
        vmovss  -144(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -204(%ebp), %eax
        vmovss  -140(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -200(%ebp), %eax
        vmovss  -136(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -196(%ebp), %eax
        vmovss  -132(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -192(%ebp), %eax
        vmovss  -128(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -188(%ebp), %eax
        vmovss  -124(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -184(%ebp), %eax
        vmovss  -120(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -180(%ebp), %eax
        vmovss  -116(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -304(%ebp), %eax
        vmovss  -112(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -300(%ebp), %eax
        vmovss  -108(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -296(%ebp), %eax
        vmovss  -104(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -292(%ebp), %eax
        vmovss  -100(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -288(%ebp), %eax
        vmovss  -96(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -284(%ebp), %eax
        vmovss  -92(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -280(%ebp), %eax
        vmovss  -88(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -276(%ebp), %eax
        vmovss  -84(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -272(%ebp), %eax
        vmovss  -80(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -268(%ebp), %eax
        vmovss  -76(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -264(%ebp), %eax
        vmovss  -72(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -260(%ebp), %eax
        vmovss  -68(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -256(%ebp), %eax
        vmovss  -64(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -252(%ebp), %eax
        vmovss  -60(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -248(%ebp), %eax
        vmovss  -56(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -244(%ebp), %eax
        vmovss  -52(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        addl    $296, %esp
        popq    %r10
        .cfi_def_cfa 10, 0
        popq    %rbp
        leal    -8(%r10), %esp
        .cfi_def_cfa 7, 8
        ret
#endif
.endm

ENTRY (_ZGVeN16vvv_sincosf_knl)
WRAPPER_AVX512_vvv_vl4l4 _ZGVeN16vl4l4_sincosf_knl
END (_ZGVeN16vvv_sincosf_knl)

ENTRY (_ZGVeN16vvv_sincosf_skx)
WRAPPER_AVX512_vvv_vl4l4 _ZGVeN16vl4l4_sincosf_skx
END (_ZGVeN16vvv_sincosf_skx)

	.section .rodata, "a"
.L_2il0floatpacket.13:
	.long	0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff
	.type	.L_2il0floatpacket.13,@object

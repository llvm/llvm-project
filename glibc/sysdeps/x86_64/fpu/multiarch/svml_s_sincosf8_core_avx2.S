/* Function sincosf vectorized with AVX2.
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

	.text
ENTRY (_ZGVdN8vl4l4_sincosf_avx2)
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

        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $448, %rsp
        movq      __svml_s_trig_data@GOTPCREL(%rip), %rax
        vmovdqa   %ymm0, %ymm5
        vmovups   %ymm13, 352(%rsp)
        vmovups __sAbsMask(%rax), %ymm2
        vmovups __sInvPI(%rax), %ymm1
        vmovups __sPI1_FMA(%rax), %ymm13
        vmovups   %ymm15, 288(%rsp)

/* Absolute argument computation */
        vandps    %ymm2, %ymm5, %ymm4

/* c) Getting octant Y by 2/Pi multiplication
   d) Add "Right Shifter" value */
        vfmadd213ps __sRShifter(%rax), %ymm4, %ymm1

/* e) Treat obtained value as integer S for destination sign setting */
        vpslld    $31, %ymm1, %ymm0

/* g) Subtract "Right Shifter" (0x4B000000) value */
        vsubps __sRShifter(%rax), %ymm1, %ymm1

/* h) Subtract Y*(PI/2) from X argument, where PI/2 divided to 3 parts:
      X = X - Y*PI1 - Y*PI2 - Y*PI3 */
        vmovdqa   %ymm4, %ymm7
        vfnmadd231ps %ymm1, %ymm13, %ymm7
        vfnmadd231ps __sPI2_FMA(%rax), %ymm1, %ymm7
        vandps __sSignMask(%rax), %ymm7, %ymm15
        vxorps __sOneHalf(%rax), %ymm15, %ymm6

/* Add correction term 0.5 for cos() part */
        vaddps    %ymm6, %ymm1, %ymm6
        vmovdqa   %ymm4, %ymm3
        vfnmadd231ps %ymm6, %ymm13, %ymm3
        vmovups __sPI3_FMA(%rax), %ymm13
        vcmpnle_uqps __sRangeReductionVal(%rax), %ymm4, %ymm4
        vfnmadd231ps __sPI2_FMA(%rax), %ymm6, %ymm3
        vfnmadd213ps %ymm7, %ymm13, %ymm1
        vfnmadd213ps %ymm3, %ymm13, %ymm6

/* Result sign calculations */
        vxorps __sSignMask(%rax), %ymm15, %ymm3
        vxorps    %ymm0, %ymm3, %ymm7
        vxorps    %ymm7, %ymm6, %ymm3
        vxorps    %ymm0, %ymm1, %ymm15
        vandnps   %ymm5, %ymm2, %ymm6
        vmovups __sA7_FMA(%rax), %ymm2
        vmulps    %ymm15, %ymm15, %ymm13
        vmovups __sA9_FMA(%rax), %ymm7
        vmulps    %ymm3, %ymm3, %ymm1

/* 2) Polynomial (minimax for sin within  [-Pi/4; +Pi/4] interval)
      a) Calculate X^2 = X * X
      b) Calculate 2 polynomials for sin and cos:
         RS = X * ( A0 + X^2 * (A1 + x^2 * (A2 + x^2 * (A3))));
         RC = B0 + X^2 * (B1 + x^2 * (B2 + x^2 * (B3 + x^2 * (B4)))) */
        vmovdqa   %ymm2, %ymm0
        vfmadd231ps __sA9_FMA(%rax), %ymm13, %ymm0
        vfmadd213ps %ymm2, %ymm1, %ymm7
        vfmadd213ps __sA5_FMA(%rax), %ymm13, %ymm0
        vfmadd213ps __sA5_FMA(%rax), %ymm1, %ymm7
        vfmadd213ps __sA3(%rax), %ymm13, %ymm0
        vfmadd213ps __sA3(%rax), %ymm1, %ymm7
        vmulps    %ymm13, %ymm0, %ymm13
        vmulps    %ymm1, %ymm7, %ymm1
        vfmadd213ps %ymm15, %ymm15, %ymm13
        vfmadd213ps %ymm3, %ymm3, %ymm1
        vmovmskps %ymm4, %ecx
        vxorps    %ymm6, %ymm13, %ymm0
        testl     %ecx, %ecx
        jne       .LBL_1_3

.LBL_1_2:
        cfi_remember_state
        vmovups   352(%rsp), %ymm13
        vmovups   288(%rsp), %ymm15
        vmovups   %ymm0, (%rdi)
        vmovups   %ymm1, (%rsi)
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret

.LBL_1_3:
        cfi_restore_state
        vmovups   %ymm5, 256(%rsp)
        vmovups   %ymm0, 320(%rsp)
        vmovups   %ymm1, 384(%rsp)
        je        .LBL_1_2

        xorb      %dl, %dl
        xorl      %eax, %eax
        vmovups   %ymm8, 160(%rsp)
        vmovups   %ymm9, 128(%rsp)
        vmovups   %ymm10, 96(%rsp)
        vmovups   %ymm11, 64(%rsp)
        vmovups   %ymm12, 32(%rsp)
        vmovups   %ymm14, (%rsp)
        movq      %rsi, 192(%rsp)
        movq      %r12, 232(%rsp)
        cfi_offset_rel_rsp (12, 232)
        movb      %dl, %r12b
        movq      %r13, 224(%rsp)
        cfi_offset_rel_rsp (13, 224)
        movl      %eax, %r13d
        movq      %r14, 216(%rsp)
        cfi_offset_rel_rsp (14, 216)
        movl      %ecx, %r14d
        movq      %r15, 208(%rsp)
        cfi_offset_rel_rsp (14, 208)
        movq      %rbx, 200(%rsp)
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

        vmovups   160(%rsp), %ymm8
        movq      %rbx, %rdi
        vmovups   128(%rsp), %ymm9
        vmovups   96(%rsp), %ymm10
        vmovups   64(%rsp), %ymm11
        vmovups   32(%rsp), %ymm12
        vmovups   (%rsp), %ymm14
        vmovups   320(%rsp), %ymm0
        vmovups   384(%rsp), %ymm1
        movq      192(%rsp), %rsi
        movq      232(%rsp), %r12
        cfi_restore (%r12)
        movq      224(%rsp), %r13
        cfi_restore (%r13)
        movq      216(%rsp), %r14
        cfi_restore (%r14)
        movq      208(%rsp), %r15
        cfi_restore (%r15)
        movq      200(%rsp), %rbx
        jmp       .LBL_1_2

.LBL_1_10:
        cfi_restore_state
        movzbl    %r12b, %r15d
        vmovss    260(%rsp,%r15,8), %xmm0
        vzeroupper

        call      JUMPTARGET(sinf)

        vmovss    %xmm0, 324(%rsp,%r15,8)
        vmovss    260(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(cosf)

        vmovss    %xmm0, 388(%rsp,%r15,8)
        jmp       .LBL_1_8

.LBL_1_13:
        movzbl    %r12b, %r15d
        vmovss    256(%rsp,%r15,8), %xmm0
        vzeroupper

        call      JUMPTARGET(sinf)

        vmovss    %xmm0, 320(%rsp,%r15,8)
        vmovss    256(%rsp,%r15,8), %xmm0

        call      JUMPTARGET(cosf)

        vmovss    %xmm0, 384(%rsp,%r15,8)
        jmp       .LBL_1_7

END (_ZGVdN8vl4l4_sincosf_avx2)
libmvec_hidden_def(_ZGVdN8vl4l4_sincosf_avx2)

/* vvv version implemented with wrapper to vl4l4 variant.  */
ENTRY (_ZGVdN8vvv_sincosf_avx2)
#ifndef __ILP32__
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-32, %rsp
        subq      $192, %rsp
        vmovdqu   %ymm1, 64(%rsp)
        lea       (%rsp), %rdi
        vmovdqu   %ymm2, 96(%rdi)
        vmovdqu   %ymm3, 128(%rdi)
        vmovdqu   %ymm4, 160(%rdi)
        lea       32(%rsp), %rsi
        call      HIDDEN_JUMPTARGET(_ZGVdN8vl4l4_sincosf_avx2)
        movq      64(%rsp), %rdx
        movq      72(%rsp), %rsi
        movq      80(%rsp), %r8
        movq      88(%rsp), %r10
        movl      (%rsp), %eax
        movl      4(%rsp), %ecx
        movl      8(%rsp), %edi
        movl      12(%rsp), %r9d
        movl      %eax, (%rdx)
        movl      %ecx, (%rsi)
        movq      96(%rsp), %rax
        movq      104(%rsp), %rcx
        movl      %edi, (%r8)
        movl      %r9d, (%r10)
        movq      112(%rsp), %rdi
        movq      120(%rsp), %r9
        movl      16(%rsp), %r11d
        movl      20(%rsp), %edx
        movl      24(%rsp), %esi
        movl      28(%rsp), %r8d
        movl      %r11d, (%rax)
        movl      %edx, (%rcx)
        movq      128(%rsp), %r11
        movq      136(%rsp), %rdx
        movl      %esi, (%rdi)
        movl      %r8d, (%r9)
        movq      144(%rsp), %rsi
        movq      152(%rsp), %r8
        movl      32(%rsp), %r10d
        movl      36(%rsp), %eax
        movl      40(%rsp), %ecx
        movl      44(%rsp), %edi
        movl      %r10d, (%r11)
        movl      %eax, (%rdx)
        movq      160(%rsp), %r10
        movq      168(%rsp), %rax
        movl      %ecx, (%rsi)
        movl      %edi, (%r8)
        movq      176(%rsp), %rcx
        movq      184(%rsp), %rdi
        movl      48(%rsp), %r9d
        movl      52(%rsp), %r11d
        movl      56(%rsp), %edx
        movl      60(%rsp), %esi
        movl      %r9d, (%r10)
        movl      %r11d, (%rax)
        movl      %edx, (%rcx)
        movl      %esi, (%rdi)
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
        subl    $136, %esp
        vmovdqa %ymm1, -112(%ebp)
        vmovdqa %ymm2, -144(%ebp)
        call    HIDDEN_JUMPTARGET(_ZGVdN8vl4l4_sincosf_avx2)
        vmovdqa -112(%ebp), %xmm0
        vmovq   %xmm0, %rax
        vmovss  -80(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        vmovss  -76(%ebp), %xmm0
        shrq    $32, %rax
        vmovss  %xmm0, (%eax)
        movq    -104(%ebp), %rax
        vmovss  -72(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        vmovss  -68(%ebp), %xmm0
        shrq    $32, %rax
        vmovss  %xmm0, (%eax)
        movq    -96(%ebp), %rax
        vmovss  -64(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        vmovss  -60(%ebp), %xmm0
        shrq    $32, %rax
        vmovss  %xmm0, (%eax)
        movq    -88(%ebp), %rax
        vmovss  -56(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        vmovss  -52(%ebp), %xmm0
        shrq    $32, %rax
        vmovss  %xmm0, (%eax)
        vmovdqa -144(%ebp), %xmm0
        vmovq   %xmm0, %rax
        vmovss  -48(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        vmovss  -44(%ebp), %xmm0
        shrq    $32, %rax
        vmovss  %xmm0, (%eax)
        movq    -136(%ebp), %rax
        vmovss  -40(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        vmovss  -36(%ebp), %xmm0
        shrq    $32, %rax
        vmovss  %xmm0, (%eax)
        movq    -128(%ebp), %rax
        vmovss  -32(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        vmovss  -28(%ebp), %xmm0
        shrq    $32, %rax
        vmovss  %xmm0, (%eax)
        movq    -120(%ebp), %rax
        vmovss  -24(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        vmovss  -20(%ebp), %xmm0
        shrq    $32, %rax
        vmovss  %xmm0, (%eax)
        addl    $136, %esp
        popq    %r10
        .cfi_def_cfa 10, 0
        popq    %rbp
        leal    -8(%r10), %esp
        .cfi_def_cfa 7, 8
        ret
#endif
END (_ZGVdN8vvv_sincosf_avx2)

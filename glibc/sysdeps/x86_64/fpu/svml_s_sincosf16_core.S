/* Function sincosf vectorized with AVX-512. Wrapper to AVX2 version.
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
#include "svml_s_wrapper_impl.h"

	.text
ENTRY (_ZGVeN16vl4l4_sincosf)
WRAPPER_IMPL_AVX512_fFF _ZGVdN8vl4l4_sincosf
END (_ZGVeN16vl4l4_sincosf)

/* AVX512 ISA version as wrapper to AVX2 ISA version (for vector
   function declared with #pragma omp declare simd notinbranch).  */
.macro WRAPPER_IMPL_AVX512_fFF_vvv callee
#ifndef __ILP32__
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $448, %rsp
        vmovups   %zmm0, 384(%rsp)
        lea       (%rsp), %rdi
        vmovups   %zmm1, 128(%rdi)
        vmovups   %zmm2, 192(%rdi)
        vmovups   %zmm3, 256(%rdi)
        vmovups   %zmm4, 320(%rdi)
        lea       64(%rsp), %rsi
        call      HIDDEN_JUMPTARGET(\callee)
        vmovdqu   416(%rsp), %ymm0
        lea       32(%rsp), %rdi
        lea       96(%rsp), %rsi
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
        pushq   %r12
        leal    -112(%rbp), %esi
        pushq   %r10
        .cfi_escape 0xf,0x3,0x76,0x70,0x6
        .cfi_escape 0x10,0xc,0x2,0x76,0x78
        leal    -176(%rbp), %edi
        movq    %rsi, %r12
        pushq   %rbx
        .cfi_escape 0x10,0x3,0x2,0x76,0x68
        movq    %rdi, %rbx
        subl    $344, %esp
        vmovdqa64 %zmm1, -240(%ebp)
        vmovdqa64 %zmm2, -304(%ebp)
        vmovaps   %zmm0, -368(%ebp)
        call    HIDDEN_JUMPTARGET(\callee)
        leal    32(%r12), %esi
        vmovups -336(%ebp), %ymm0
        leal    32(%rbx), %edi
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
        addl    $344, %esp
        popq    %rbx
        popq    %r10
        .cfi_def_cfa 10, 0
        popq    %r12
        popq    %rbp
        leal    -8(%r10), %esp
        .cfi_def_cfa 7, 8
        ret
#endif
.endm

ENTRY (_ZGVeN16vvv_sincosf)
WRAPPER_IMPL_AVX512_fFF_vvv _ZGVdN8vl4l4_sincosf
END (_ZGVeN16vvv_sincosf)

/* Function sincos vectorized with AVX-512. Wrapper to AVX2 version.
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
#include "svml_d_wrapper_impl.h"

	.text
ENTRY (_ZGVeN8vl8l8_sincos)
WRAPPER_IMPL_AVX512_fFF _ZGVdN4vl8l8_sincos
END (_ZGVeN8vl8l8_sincos)

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
        subq      $320, %rsp
        vmovups    %zmm0, 256(%rsp)
        lea       (%rsp), %rdi
        vmovups   %zmm1, 128(%rdi)
        vmovups   %zmm2, 192(%rdi)
        lea       64(%rsp), %rsi
        call      HIDDEN_JUMPTARGET(\callee)
        vmovdqu   288(%rsp), %ymm0
        lea       32(%rsp), %rdi
        lea       96(%rsp), %rsi
        call      HIDDEN_JUMPTARGET(\callee)
        movq      128(%rsp), %rdx
        movq      192(%rsp), %rsi
        movq      136(%rsp), %r8
        movq      200(%rsp), %r10
        movq      (%rsp), %rax
        movq      64(%rsp), %rcx
        movq      8(%rsp), %rdi
        movq      72(%rsp), %r9
        movq      %rax, (%rdx)
        movq      %rcx, (%rsi)
        movq      144(%rsp), %rax
        movq      208(%rsp), %rcx
        movq      %rdi, (%r8)
        movq      %r9, (%r10)
        movq      152(%rsp), %rdi
        movq      216(%rsp), %r9
        movq      16(%rsp), %r11
        movq      80(%rsp), %rdx
        movq      24(%rsp), %rsi
        movq      88(%rsp), %r8
        movq      %r11, (%rax)
        movq      %rdx, (%rcx)
        movq      160(%rsp), %r11
        movq      224(%rsp), %rdx
        movq      %rsi, (%rdi)
        movq      %r8, (%r9)
        movq      168(%rsp), %rsi
        movq      232(%rsp), %r8
        movq      32(%rsp), %r10
        movq      96(%rsp), %rax
        movq      40(%rsp), %rcx
        movq      104(%rsp), %rdi
        movq      %r10, (%r11)
        movq      %rax, (%rdx)
        movq      176(%rsp), %r10
        movq      240(%rsp), %rax
        movq      %rcx, (%rsi)
        movq      %rdi, (%r8)
        movq      184(%rsp), %rcx
        movq      248(%rsp), %rdi
        movq      48(%rsp), %r9
        movq      112(%rsp), %r11
        movq      56(%rsp), %rdx
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
        subl    $280, %esp
        vmovdqa %ymm1, -208(%ebp)
        vmovdqa %ymm2, -240(%ebp)
        vmovapd %zmm0, -304(%ebp)
        call    HIDDEN_JUMPTARGET(\callee)
        leal    32(%r12), %esi
        vmovupd -272(%ebp), %ymm0
        leal    32(%rbx), %edi
        call    HIDDEN_JUMPTARGET(\callee)
        movl    -208(%ebp), %eax
        vmovsd  -176(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -204(%ebp), %eax
        vmovsd  -168(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -200(%ebp), %eax
        vmovsd  -160(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -196(%ebp), %eax
        vmovsd  -152(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -192(%ebp), %eax
        vmovsd  -144(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -188(%ebp), %eax
        vmovsd  -136(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -184(%ebp), %eax
        vmovsd  -128(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -180(%ebp), %eax
        vmovsd  -120(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -240(%ebp), %eax
        vmovsd  -112(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -236(%ebp), %eax
        vmovsd  -104(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -232(%ebp), %eax
        vmovsd  -96(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -228(%ebp), %eax
        vmovsd  -88(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -224(%ebp), %eax
        vmovsd  -80(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -220(%ebp), %eax
        vmovsd  -72(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -216(%ebp), %eax
        vmovsd  -64(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        movl    -212(%ebp), %eax
        vmovsd  -56(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        addl    $280, %esp
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

ENTRY (_ZGVeN8vvv_sincos)
WRAPPER_IMPL_AVX512_fFF_vvv _ZGVdN4vl8l8_sincos
END (_ZGVeN8vvv_sincos)

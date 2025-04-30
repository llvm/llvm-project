/* Function sincosf vectorized in AVX ISA as wrapper to SSE4 ISA version.
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
ENTRY (_ZGVcN8vl4l4_sincosf)
WRAPPER_IMPL_AVX_fFF _ZGVbN4vl4l4_sincosf
END (_ZGVcN8vl4l4_sincosf)

/* AVX ISA version as wrapper to SSE ISA version (for vector
   function declared with #pragma omp declare simd notinbranch).  */
.macro WRAPPER_IMPL_AVX_fFF_vvv callee
#ifndef __ILP32__
        pushq     %rbp
        movq      %rsp, %rbp
        andq      $-32, %rsp
        subq      $224, %rsp
        vmovups   %ymm0, 64(%rsp)
        lea       (%rsp), %rdi
        vmovdqu   %xmm1, 96(%rdi)
        vmovdqu   %xmm2, 112(%rdi)
        vmovdqu   %xmm3, 128(%rdi)
        vmovdqu   %xmm4, 144(%rdi)
        vmovdqu   %xmm5, 160(%rdi)
        lea       32(%rsp), %rsi
        vmovdqu   %xmm6, 144(%rsi)
        vmovdqu   %xmm7, 160(%rsi)
        vzeroupper
        call      HIDDEN_JUMPTARGET(\callee)
        vmovdqu   80(%rsp), %xmm0
        lea       16(%rsp), %rdi
        lea       48(%rsp), %rsi
        call      HIDDEN_JUMPTARGET(\callee)
        movq      96(%rsp), %rdx
        movq      104(%rsp), %rsi
        movq      112(%rsp), %r8
        movq      120(%rsp), %r10
        movl      (%rsp), %eax
        movl      4(%rsp), %ecx
        movl      8(%rsp), %edi
        movl      12(%rsp), %r9d
        movl      %eax, (%rdx)
        movl      %ecx, (%rsi)
        movq      128(%rsp), %rax
        movq      136(%rsp), %rcx
        movl      %edi, (%r8)
        movl      %r9d, (%r10)
        movq      144(%rsp), %rdi
        movq      152(%rsp), %r9
        movl      16(%rsp), %r11d
        movl      20(%rsp), %edx
        movl      24(%rsp), %esi
        movl      28(%rsp), %r8d
        movl      %r11d, (%rax)
        movl      %edx, (%rcx)
        movq      160(%rsp), %r11
        movq      168(%rsp), %rdx
        movl      %esi, (%rdi)
        movl      %r8d, (%r9)
        movq      176(%rsp), %rsi
        movq      184(%rsp), %r8
        movl      32(%rsp), %r10d
        movl      36(%rsp), %eax
        movl      40(%rsp), %ecx
        movl      44(%rsp), %edi
        movl      %r10d, (%r11)
        movl      %eax, (%rdx)
        movq      192(%rsp), %r10
        movq      200(%rsp), %rax
        movl      %ecx, (%rsi)
        movl      %edi, (%r8)
        movq      16(%rbp), %rcx
        movq      24(%rbp), %rdi
        movl      48(%rsp), %r9d
        movl      52(%rsp), %r11d
        movl      56(%rsp), %edx
        movl      60(%rsp), %esi
        movl      %r9d, (%r10)
        movl      %r11d, (%rax)
        movl      %edx, (%rcx)
        movl      %esi, (%rdi)
        movq      %rbp, %rsp
        popq      %rbp
        ret
#else
        leal    8(%rsp), %r10d
        .cfi_def_cfa 10, 0
        andl    $-32, %esp
        pushq   -8(%r10d)
        pushq   %rbp
        .cfi_escape 0x10,0x6,0x2,0x76,0
        movl    %esp, %ebp
        pushq   %r12
        leal    -80(%rbp), %esi
        pushq   %r10
        .cfi_escape 0xf,0x3,0x76,0x70,0x6
        .cfi_escape 0x10,0xc,0x2,0x76,0x78
        leal    -112(%rbp), %edi
        movq    %rsi, %r12
        pushq   %rbx
        .cfi_escape 0x10,0x3,0x2,0x76,0x68
        movq    %rdi, %rbx
        subl    $184, %esp
        vmovaps %xmm1, -128(%ebp)
        vmovaps %xmm2, -144(%ebp)
        vmovaps %xmm3, -160(%ebp)
        vmovaps %xmm4, -176(%ebp)
        vmovaps %ymm0, -208(%ebp)
        vzeroupper
        call    HIDDEN_JUMPTARGET(\callee)
        leal    16(%r12), %esi
        vmovups -192(%ebp), %xmm0
        leal    16(%rbx), %edi
        call    HIDDEN_JUMPTARGET(\callee)
        movq    -128(%ebp), %rax
        vmovss  -112(%ebp), %xmm0
        vmovdqa -128(%ebp), %xmm7
        vmovdqa -144(%ebp), %xmm3
        vmovss  %xmm0, (%eax)
        vmovss  -108(%ebp), %xmm0
        vpextrd $1, %xmm7, %eax
        vmovss  %xmm0, (%eax)
        movq    -120(%ebp), %rax
        vmovss  -104(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        vmovss  -100(%ebp), %xmm0
        vpextrd $3, %xmm7, %eax
        vmovdqa -160(%ebp), %xmm7
        vmovss  %xmm0, (%eax)
        movq    -144(%ebp), %rax
        vmovss  -96(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        vmovss  -92(%ebp), %xmm0
        vpextrd $1, %xmm3, %eax
        vmovss  %xmm0, (%eax)
        movq    -136(%ebp), %rax
        vmovss  -88(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        vmovss  -84(%ebp), %xmm0
        vpextrd $3, %xmm3, %eax
        vmovss  %xmm0, (%eax)
        movq    -160(%ebp), %rax
        vmovss  -80(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        vmovss  -76(%ebp), %xmm0
        vpextrd $1, %xmm7, %eax
        vmovss  %xmm0, (%eax)
        movq    -152(%ebp), %rax
        vmovss  -72(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        vmovss  -68(%ebp), %xmm0
        vpextrd $3, %xmm7, %eax
        vmovss  %xmm0, (%eax)
        movq    -176(%ebp), %rax
        vmovss  -64(%ebp), %xmm0
        vmovdqa -176(%ebp), %xmm3
        vmovss  %xmm0, (%eax)
        vmovss  -60(%ebp), %xmm0
        vpextrd $1, %xmm3, %eax
        vmovss  %xmm0, (%eax)
        movq    -168(%ebp), %rax
        vmovss  -56(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        vmovss  -52(%ebp), %xmm0
        vpextrd $3, %xmm3, %eax
        vmovss  %xmm0, (%eax)
        addl    $184, %esp
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

ENTRY (_ZGVcN8vvv_sincosf)
WRAPPER_IMPL_AVX_fFF_vvv _ZGVbN4vl4l4_sincosf
END (_ZGVcN8vvv_sincosf)

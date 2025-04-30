/* Function sincosf vectorized with AVX2, wrapper version.
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
ENTRY (_ZGVdN8vl4l4_sincosf)
WRAPPER_IMPL_AVX_fFF _ZGVbN4vl4l4_sincosf
END (_ZGVdN8vl4l4_sincosf)
libmvec_hidden_def (_ZGVdN8vl4l4_sincosf)

/* AVX2 ISA version as wrapper to SSE ISA version (for vector
   function declared with #pragma omp declare simd notinbranch).  */
.macro WRAPPER_IMPL_AVX2_fFF_vvv callee
#ifndef __ILP32__
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-32, %rsp
        subq      $224, %rsp
        vmovups   %ymm0, 192(%rsp)
        lea       (%rsp), %rdi
        vmovdqu   %ymm1, 64(%rdi)
        vmovdqu   %ymm2, 96(%rdi)
        vmovdqu   %ymm3, 128(%rdi)
        vmovdqu   %ymm4, 160(%rdi)
        lea       32(%rsp), %rsi
	vzeroupper
        call      HIDDEN_JUMPTARGET(\callee)
        vmovups   208(%rsp), %xmm0
        lea       16(%rsp), %rdi
        lea       48(%rsp), %rsi
        call      HIDDEN_JUMPTARGET(\callee)
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
        vmovdqa %ymm1, -144(%ebp)
        vmovdqa %ymm2, -176(%ebp)
        vmovaps %ymm0, -208(%ebp)
	vzeroupper
        call    HIDDEN_JUMPTARGET(\callee)
        leal    16(%r12), %esi
        vmovups -192(%ebp), %xmm0
        leal    16(%rbx), %edi
        call    HIDDEN_JUMPTARGET(\callee)
        movl    -144(%ebp), %eax
        vmovss  -112(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -140(%ebp), %eax
        vmovss  -108(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -136(%ebp), %eax
        vmovss  -104(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -132(%ebp), %eax
        vmovss  -100(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -128(%ebp), %eax
        vmovss  -96(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -124(%ebp), %eax
        vmovss  -92(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -120(%ebp), %eax
        vmovss  -88(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -116(%ebp), %eax
        vmovss  -84(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -176(%ebp), %eax
        vmovss  -80(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -172(%ebp), %eax
        vmovss  -76(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -168(%ebp), %eax
        vmovss  -72(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -164(%ebp), %eax
        vmovss  -68(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -160(%ebp), %eax
        vmovss  -64(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -156(%ebp), %eax
        vmovss  -60(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -152(%ebp), %eax
        vmovss  -56(%ebp), %xmm0
        vmovss  %xmm0, (%eax)
        movl    -148(%ebp), %eax
        vmovss  -52(%ebp), %xmm0
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

ENTRY (_ZGVdN8vvv_sincosf)
WRAPPER_IMPL_AVX2_fFF_vvv _ZGVbN4vl4l4_sincosf
END (_ZGVdN8vvv_sincosf)

#ifndef USE_MULTIARCH
 libmvec_hidden_def (_ZGVdN8vvv_sincosf)
#endif

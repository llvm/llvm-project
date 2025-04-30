/* Function sincos vectorized with AVX2, wrapper version.
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
ENTRY (_ZGVdN4vl8l8_sincos)
WRAPPER_IMPL_AVX_fFF _ZGVbN2vl8l8_sincos
END (_ZGVdN4vl8l8_sincos)
libmvec_hidden_def (_ZGVdN4vl8l8_sincos)

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
        subq      $160, %rsp
        vmovupd   %ymm0, 128(%rsp)
        lea       (%rsp), %rdi
        vmovdqu   %ymm1, 64(%rdi)
        vmovdqu   %ymm2, 96(%rdi)
        lea       32(%rsp), %rsi
        vzeroupper
        call      HIDDEN_JUMPTARGET(\callee)
        vmovupd   144(%rsp), %xmm0
        lea       16(%rsp), %rdi
        lea       48(%rsp), %rsi
        call      HIDDEN_JUMPTARGET(\callee)
        movq      64(%rsp), %rdx
        movq      96(%rsp), %rsi
        movq      72(%rsp), %r8
        movq      104(%rsp), %r10
        movq      (%rsp), %rax
        movq      32(%rsp), %rcx
        movq      8(%rsp), %rdi
        movq      40(%rsp), %r9
        movq      %rax, (%rdx)
        movq      %rcx, (%rsi)
        movq      80(%rsp), %rax
        movq      112(%rsp), %rcx
        movq      %rdi, (%r8)
        movq      %r9, (%r10)
        movq      88(%rsp), %rdi
        movq      120(%rsp), %r9
        movq      16(%rsp), %r11
        movq      48(%rsp), %rdx
        movq      24(%rsp), %rsi
        movq      56(%rsp), %r8
        movq      %r11, (%rax)
        movq      %rdx, (%rcx)
        movq      %rsi, (%rdi)
        movq      %r8, (%r9)
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
        subl    $152, %esp
        vmovaps %xmm1, -128(%ebp)
        vmovaps %xmm2, -144(%ebp)
        vmovapd %ymm0, -176(%ebp)
        vzeroupper
        call    HIDDEN_JUMPTARGET(\callee)
        leal    16(%r12), %esi
        vmovapd -160(%ebp), %xmm0
        leal    16(%rbx), %edi
        call    HIDDEN_JUMPTARGET(\callee)
        movq    -128(%ebp), %rax
        vmovsd  -112(%ebp), %xmm0
        vmovdqa -128(%ebp), %xmm5
        vmovdqa -144(%ebp), %xmm1
        vmovsd  %xmm0, (%eax)
        vmovsd  -104(%ebp), %xmm0
        vpextrd $1, %xmm5, %eax
        vmovsd  %xmm0, (%eax)
        movq    -120(%ebp), %rax
        vmovsd  -96(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        vmovsd  -88(%ebp), %xmm0
        vpextrd $3, %xmm5, %eax
        vmovsd  %xmm0, (%eax)
        movq    -144(%ebp), %rax
        vmovsd  -80(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        vmovsd  -72(%ebp), %xmm0
        vpextrd $1, %xmm1, %eax
        vmovsd  %xmm0, (%eax)
        movq    -136(%ebp), %rax
        vmovsd  -64(%ebp), %xmm0
        vmovsd  %xmm0, (%eax)
        vmovsd  -56(%ebp), %xmm0
        vpextrd $3, %xmm1, %eax
        vmovsd  %xmm0, (%eax)
        addl    $152, %esp
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

ENTRY (_ZGVdN4vvv_sincos)
WRAPPER_IMPL_AVX2_fFF_vvv _ZGVbN2vl8l8_sincos
END (_ZGVdN4vvv_sincos)

#ifndef USE_MULTIARCH
 libmvec_hidden_def (_ZGVdN4vvv_sincos)
#endif

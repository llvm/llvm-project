/* Function sincos vectorized with SSE2.
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
ENTRY (_ZGVbN2vl8l8_sincos)
WRAPPER_IMPL_SSE2_fFF sincos
END (_ZGVbN2vl8l8_sincos)
libmvec_hidden_def (_ZGVbN2vl8l8_sincos)

/* SSE2 ISA version as wrapper to scalar (for vector
   function declared with #pragma omp declare simd notinbranch).  */
.macro WRAPPER_IMPL_SSE2_fFF_vvv callee
#ifndef __ILP32__
        subq      $88, %rsp
        cfi_adjust_cfa_offset(88)
        movaps    %xmm0, 64(%rsp)
        lea       (%rsp), %rdi
        movdqa    %xmm1, 32(%rdi)
        lea       16(%rsp), %rsi
        movdqa    %xmm2, 32(%rsi)
        call      JUMPTARGET(\callee)
        movsd     72(%rsp), %xmm0
        lea       8(%rsp), %rdi
        lea       24(%rsp), %rsi
        call      JUMPTARGET(\callee)
        movq      32(%rsp), %rdx
        movq      48(%rsp), %rsi
        movq      40(%rsp), %r8
        movq      56(%rsp), %r10
        movq      (%rsp), %rax
        movq      16(%rsp), %rcx
        movq      8(%rsp), %rdi
        movq      24(%rsp), %r9
        movq      %rax, (%rdx)
        movq      %rcx, (%rsi)
        movq      %rdi, (%r8)
        movq      %r9, (%r10)
        addq      $88, %rsp
        cfi_adjust_cfa_offset(-88)
        ret
#else
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset 6, -16
        pushq   %rbx
        .cfi_def_cfa_offset 24
        .cfi_offset 3, -24
        subl    $88, %esp
        .cfi_def_cfa_offset 112
        leal    64(%rsp), %esi
        movaps  %xmm1, 32(%esp)
        leal    48(%rsp), %edi
        movaps  %xmm2, 16(%esp)
        movq    %rsi, %rbp
        movq    %rdi, %rbx
        movaps  %xmm0, (%esp)
        call    JUMPTARGET(\callee)
        movupd  8(%esp), %xmm0
        leal    8(%rbp), %esi
        leal    8(%rbx), %edi
        call    JUMPTARGET(\callee)
        movdqa  32(%esp), %xmm1
        movsd   48(%esp), %xmm0
        movq    %xmm1, %rax
        movdqa  16(%esp), %xmm2
        movsd   %xmm0, (%eax)
        movsd   56(%esp), %xmm0
        pextrd  $1, %xmm1, %eax
        movsd   %xmm0, (%eax)
        movsd   64(%esp), %xmm0
        movq    %xmm2, %rax
        movsd   %xmm0, (%eax)
        movsd   72(%esp), %xmm0
        pextrd  $1, %xmm2, %eax
        movsd   %xmm0, (%eax)
        addl    $88, %esp
        .cfi_def_cfa_offset 24
        popq    %rbx
        .cfi_def_cfa_offset 16
        popq    %rbp
        .cfi_def_cfa_offset 8
        ret
#endif
.endm

ENTRY (_ZGVbN2vvv_sincos)
WRAPPER_IMPL_SSE2_fFF_vvv sincos
END (_ZGVbN2vvv_sincos)

#ifndef USE_MULTIARCH
 libmvec_hidden_def (_ZGVbN2vvv_sincos)
#endif

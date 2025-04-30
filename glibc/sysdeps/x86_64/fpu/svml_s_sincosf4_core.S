/* Function sincosf vectorized with SSE2.
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
ENTRY (_ZGVbN4vl4l4_sincosf)
WRAPPER_IMPL_SSE2_fFF sincosf
END (_ZGVbN4vl4l4_sincosf)
libmvec_hidden_def (_ZGVbN4vl4l4_sincosf)

/* SSE2 ISA version as wrapper to scalar (for vector
   function declared with #pragma omp declare simd notinbranch).  */
.macro WRAPPER_IMPL_SSE2_fFF_vvv callee
#ifndef __ILP32__
        subq      $120, %rsp
        cfi_adjust_cfa_offset(120)
        movaps    %xmm0, 96(%rsp)
        lea       (%rsp), %rdi
        movdqa    %xmm1, 32(%rdi)
        lea       16(%rsp), %rsi
        movdqa    %xmm2, 32(%rsi)
        movdqa    %xmm3, 48(%rsi)
        movdqa    %xmm4, 64(%rsi)
        call      JUMPTARGET(\callee)
        movss     100(%rsp), %xmm0
        lea       4(%rsp), %rdi
        lea       20(%rsp), %rsi
        call      JUMPTARGET(\callee)
        movss     104(%rsp), %xmm0
        lea       8(%rsp), %rdi
        lea       24(%rsp), %rsi
        call      JUMPTARGET(\callee)
        movss     108(%rsp), %xmm0
        lea       12(%rsp), %rdi
        lea       28(%rsp), %rsi
        call      JUMPTARGET(\callee)
        movq      32(%rsp), %rdx
        movq      40(%rsp), %rsi
        movq      48(%rsp), %r8
        movq      56(%rsp), %r10
        movl      (%rsp), %eax
        movl      4(%rsp), %ecx
        movl      8(%rsp), %edi
        movl      12(%rsp), %r9d
        movl      %eax, (%rdx)
        movl      %ecx, (%rsi)
        movq      64(%rsp), %rax
        movq      72(%rsp), %rcx
        movl      %edi, (%r8)
        movl      %r9d, (%r10)
        movq      80(%rsp), %rdi
        movq      88(%rsp), %r9
        movl      16(%rsp), %r11d
        movl      20(%rsp), %edx
        movl      24(%rsp), %esi
        movl      28(%rsp), %r8d
        movl      %r11d, (%rax)
        movl      %edx, (%rcx)
        movl      %esi, (%rdi)
        movl      %r8d, (%r9)
        addq      $120, %rsp
        cfi_adjust_cfa_offset(-120)
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
        movaps  %xmm1, (%esp)
        leal    48(%rsp), %edi
        movaps  %xmm2, 16(%esp)
        movq    %rsi, %rbp
        movq    %rdi, %rbx
        movaps  %xmm0, 32(%esp)
        call    JUMPTARGET(\callee)
        movups  36(%esp), %xmm0
        leal    4(%rbp), %esi
        leal    4(%rbx), %edi
        call    JUMPTARGET(\callee)
        movups  40(%esp), %xmm0
        leal    8(%rbp), %esi
        leal    8(%rbx), %edi
        call    JUMPTARGET(\callee)
        movups  44(%esp), %xmm0
        leal    12(%rbp), %esi
        leal    12(%rbx), %edi
        call    JUMPTARGET(\callee)
        movq    (%esp), %rax
        movss   48(%esp), %xmm0
        movdqa  (%esp), %xmm4
        movdqa  16(%esp), %xmm7
        movss   %xmm0, (%eax)
        movss   52(%esp), %xmm0
        pextrd  $1, %xmm4, %eax
        movss   %xmm0, (%eax)
        movq    8(%esp), %rax
        movss   56(%esp), %xmm0
        movss   %xmm0, (%eax)
        movss   60(%esp), %xmm0
        pextrd  $3, %xmm4, %eax
        movss   %xmm0, (%eax)
        movq    16(%esp), %rax
        movss   64(%esp), %xmm0
        movss   %xmm0, (%eax)
        movss   68(%esp), %xmm0
        pextrd  $1, %xmm7, %eax
        movss   %xmm0, (%eax)
        movq    24(%esp), %rax
        movss   72(%esp), %xmm0
        movss   %xmm0, (%eax)
        movss   76(%esp), %xmm0
        pextrd  $3, %xmm7, %eax
        movss   %xmm0, (%eax)
        addl    $88, %esp
        .cfi_def_cfa_offset 24
        popq    %rbx
        .cfi_def_cfa_offset 16
        popq    %rbp
        .cfi_def_cfa_offset 8
        ret
#endif
.endm

ENTRY (_ZGVbN4vvv_sincosf)
WRAPPER_IMPL_SSE2_fFF_vvv sincosf
END (_ZGVbN4vvv_sincosf)

#ifndef USE_MULTIARCH
 libmvec_hidden_def (_ZGVbN4vvv_sincosf)
#endif

/* Wrapper implementations of vector math functions.
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

/* SSE2 ISA version as wrapper to scalar.  */
.macro WRAPPER_IMPL_SSE2 callee
        subq      $40, %rsp
        cfi_adjust_cfa_offset(40)
        movaps    %xmm0, (%rsp)
        call      JUMPTARGET(\callee)
        movss     %xmm0, 16(%rsp)
        movss     4(%rsp), %xmm0
        call      JUMPTARGET(\callee)
        movss     %xmm0, 20(%rsp)
        movss     8(%rsp), %xmm0
        call      JUMPTARGET(\callee)
        movss     %xmm0, 24(%rsp)
        movss     12(%rsp), %xmm0
        call      JUMPTARGET(\callee)
        movss     16(%rsp), %xmm3
        movss     20(%rsp), %xmm2
        movss     24(%rsp), %xmm1
        movss     %xmm0, 28(%rsp)
        unpcklps  %xmm1, %xmm3
        unpcklps  %xmm0, %xmm2
        unpcklps  %xmm2, %xmm3
        movaps    %xmm3, %xmm0
        addq      $40, %rsp
        cfi_adjust_cfa_offset(-40)
        ret
.endm

/* 2 argument SSE2 ISA version as wrapper to scalar.  */
.macro WRAPPER_IMPL_SSE2_ff callee
        subq      $56, %rsp
        cfi_adjust_cfa_offset(56)
        movaps    %xmm0, (%rsp)
        movaps    %xmm1, 16(%rsp)
        call      JUMPTARGET(\callee)
        movss     %xmm0, 32(%rsp)
        movss     4(%rsp), %xmm0
        movss     20(%rsp), %xmm1
        call      JUMPTARGET(\callee)
        movss     %xmm0, 36(%rsp)
        movss     8(%rsp), %xmm0
        movss     24(%rsp), %xmm1
        call      JUMPTARGET(\callee)
        movss     %xmm0, 40(%rsp)
        movss     12(%rsp), %xmm0
        movss     28(%rsp), %xmm1
        call      JUMPTARGET(\callee)
        movss     32(%rsp), %xmm3
        movss     36(%rsp), %xmm2
        movss     40(%rsp), %xmm1
        movss     %xmm0, 44(%rsp)
        unpcklps  %xmm1, %xmm3
        unpcklps  %xmm0, %xmm2
        unpcklps  %xmm2, %xmm3
        movaps    %xmm3, %xmm0
        addq      $56, %rsp
        cfi_adjust_cfa_offset(-56)
        ret
.endm

/* 3 argument SSE2 ISA version as wrapper to scalar.  */
.macro WRAPPER_IMPL_SSE2_fFF callee
        pushq   %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        pushq   %rbx
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbx, 0)
        movq    %rdi, %rbp
        movq    %rsi, %rbx
        subq    $40, %rsp
        cfi_adjust_cfa_offset(40)
        leaq    24(%rsp), %rsi
        leaq    28(%rsp), %rdi
        movaps  %xmm0, (%rsp)
        call    JUMPTARGET(\callee)
        leaq    24(%rsp), %rsi
        leaq    28(%rsp), %rdi
        movss   28(%rsp), %xmm0
        movss   %xmm0, 0(%rbp)
        movaps  (%rsp), %xmm1
        movss   24(%rsp), %xmm0
        movss   %xmm0, (%rbx)
        movaps  %xmm1, %xmm0
        shufps  $85, %xmm1, %xmm0
        call    JUMPTARGET(\callee)
        movss   28(%rsp), %xmm0
        leaq    24(%rsp), %rsi
        movss   %xmm0, 4(%rbp)
        leaq    28(%rsp), %rdi
        movaps  (%rsp), %xmm1
        movss   24(%rsp), %xmm0
        movss   %xmm0, 4(%rbx)
        movaps  %xmm1, %xmm0
        unpckhps        %xmm1, %xmm0
        call    JUMPTARGET(\callee)
        movaps  (%rsp), %xmm1
        leaq    24(%rsp), %rsi
        leaq    28(%rsp), %rdi
        movss   28(%rsp), %xmm0
        shufps  $255, %xmm1, %xmm1
        movss   %xmm0, 8(%rbp)
        movss   24(%rsp), %xmm0
        movss   %xmm0, 8(%rbx)
        movaps  %xmm1, %xmm0
        call    JUMPTARGET(\callee)
        movss   28(%rsp), %xmm0
        movss   %xmm0, 12(%rbp)
        movss   24(%rsp), %xmm0
        movss   %xmm0, 12(%rbx)
        addq    $40, %rsp
        cfi_adjust_cfa_offset(-40)
        popq    %rbx
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbx)
        popq    %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret
.endm

/* AVX/AVX2 ISA version as wrapper to SSE ISA version.  */
.macro WRAPPER_IMPL_AVX callee
        pushq     	%rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      	%rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      	$-32, %rsp
        subq      	$32, %rsp
        vextractf128 	$1, %ymm0, (%rsp)
        vzeroupper
        call      	HIDDEN_JUMPTARGET(\callee)
        vmovaps   	%xmm0, 16(%rsp)
        vmovaps   	(%rsp), %xmm0
        call      	HIDDEN_JUMPTARGET(\callee)
        vmovaps   	%xmm0, %xmm1
        vmovaps   	16(%rsp), %xmm0
        vinsertf128 	$1, %xmm1, %ymm0, %ymm0
        movq      	%rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      	%rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret
.endm

/* 2 argument AVX/AVX2 ISA version as wrapper to SSE ISA version.  */
.macro WRAPPER_IMPL_AVX_ff callee
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-32, %rsp
        subq      $64, %rsp
        vextractf128 $1, %ymm0, 16(%rsp)
        vextractf128 $1, %ymm1, (%rsp)
        vzeroupper
        call      HIDDEN_JUMPTARGET(\callee)
        vmovaps   %xmm0, 32(%rsp)
        vmovaps   16(%rsp), %xmm0
        vmovaps   (%rsp), %xmm1
        call      HIDDEN_JUMPTARGET(\callee)
        vmovaps   %xmm0, %xmm1
        vmovaps   32(%rsp), %xmm0
        vinsertf128 $1, %xmm1, %ymm0, %ymm0
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret
.endm

/* 3 argument AVX/AVX2 ISA version as wrapper to SSE ISA version.  */
.macro WRAPPER_IMPL_AVX_fFF callee
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-32, %rsp
        pushq     %r13
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%r13, 0)
        pushq     %r14
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%r14, 0)
        subq      $48, %rsp
        movq      %rsi, %r14
        vmovaps   %ymm0, (%rsp)
        movq      %rdi, %r13
        vmovaps   16(%rsp), %xmm1
        vmovaps   %xmm1, 32(%rsp)
        vzeroupper
        vmovaps   (%rsp), %xmm0
        call      HIDDEN_JUMPTARGET(\callee)
        vmovaps   32(%rsp), %xmm0
        lea       (%rsp), %rdi
        lea       16(%rsp), %rsi
        call      HIDDEN_JUMPTARGET(\callee)
        vmovaps   (%rsp), %xmm0
        vmovaps   16(%rsp), %xmm1
        vmovaps   %xmm0, 16(%r13)
        vmovaps   %xmm1, 16(%r14)
        addq      $48, %rsp
        popq      %r14
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%r14)
        popq      %r13
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%r13)
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret
.endm

/* AVX512 ISA version as wrapper to AVX2 ISA version.  */
.macro WRAPPER_IMPL_AVX512 callee
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $128, %rsp
        vmovups   %zmm0, (%rsp)
        vmovupd   (%rsp), %ymm0
        call      HIDDEN_JUMPTARGET(\callee)
        vmovupd   %ymm0, 64(%rsp)
        vmovupd   32(%rsp), %ymm0
        call      HIDDEN_JUMPTARGET(\callee)
        vmovupd   %ymm0, 96(%rsp)
        vmovups   64(%rsp), %zmm0
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret
.endm

/* 2 argument AVX512 ISA version as wrapper to AVX2 ISA version.  */
.macro WRAPPER_IMPL_AVX512_ff callee
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq      %rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   %zmm0, (%rsp)
        vmovups   %zmm1, 64(%rsp)
        vmovups   (%rsp), %ymm0
        vmovups   64(%rsp), %ymm1
        call      HIDDEN_JUMPTARGET(\callee)
        vmovups   %ymm0, 128(%rsp)
        vmovups   32(%rsp), %ymm0
        vmovups   96(%rsp), %ymm1
        call      HIDDEN_JUMPTARGET(\callee)
        vmovups   %ymm0, 160(%rsp)
        vmovups   128(%rsp), %zmm0
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq      %rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret
.endm

/* 3 argument AVX512 ISA version as wrapper to AVX2 ISA version.  */
.macro WRAPPER_IMPL_AVX512_fFF callee
        pushq     %rbp
        cfi_adjust_cfa_offset (8)
        cfi_rel_offset (%rbp, 0)
        movq	%rsp, %rbp
        cfi_def_cfa_register (%rbp)
        andq      $-64, %rsp
        pushq     %r12
        pushq     %r13
        subq      $176, %rsp
        movq      %rsi, %r13
        vmovaps   %zmm0, (%rsp)
        movq      %rdi, %r12
        vmovaps   (%rsp), %ymm0
        call      HIDDEN_JUMPTARGET(\callee)
        vmovaps   32(%rsp), %ymm0
        lea       64(%rsp), %rdi
        lea       96(%rsp), %rsi
        call      HIDDEN_JUMPTARGET(\callee)
        vmovaps   64(%rsp), %ymm0
        vmovaps   96(%rsp), %ymm1
        vmovaps   %ymm0, 32(%r12)
        vmovaps   %ymm1, 32(%r13)
        addq      $176, %rsp
        popq      %r13
        popq      %r12
        movq      %rbp, %rsp
        cfi_def_cfa_register (%rsp)
        popq	%rbp
        cfi_adjust_cfa_offset (-8)
        cfi_restore (%rbp)
        ret
.endm

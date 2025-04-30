/* memset optimized with AVX512 for KNL hardware.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#if IS_IN (libc)

#include "asm-syntax.h"
#ifndef MEMSET
# define MEMSET __memset_avx512_no_vzeroupper
# define MEMSET_CHK __memset_chk_avx512_no_vzeroupper
#endif

	.section .text.avx512,"ax",@progbits
#if defined PIC
ENTRY (MEMSET_CHK)
	cmp	%RDX_LP, %RCX_LP
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END (MEMSET_CHK)
#endif

ENTRY (MEMSET)
# ifdef __ILP32__
	/* Clear the upper 32 bits.  */
	mov	%edx, %edx
# endif
	vpxor	%xmm0, %xmm0, %xmm0
	vmovd	%esi, %xmm1
	lea	(%rdi, %rdx), %rsi
	mov	%rdi, %rax
	vpshufb	%xmm0, %xmm1, %xmm0
	cmp	$16, %rdx
	jb	L(less_16bytes)
	cmp	$512, %rdx
	vbroadcastss	%xmm0, %zmm2
	ja	L(512bytesormore)
	cmp	$256, %rdx
	jb	L(less_256bytes)
	vmovups	%zmm2, (%rdi)
	vmovups %zmm2, 0x40(%rdi)
	vmovups %zmm2, 0x80(%rdi)
	vmovups %zmm2, 0xC0(%rdi)
	vmovups %zmm2, -0x100(%rsi)
	vmovups %zmm2, -0xC0(%rsi)
	vmovups %zmm2, -0x80(%rsi)
	vmovups %zmm2, -0x40(%rsi)
	ret

L(less_256bytes):
	cmp	$128, %dl
	jb	L(less_128bytes)
	vmovups	%zmm2, (%rdi)
	vmovups %zmm2, 0x40(%rdi)
	vmovups %zmm2, -0x80(%rsi)
	vmovups %zmm2, -0x40(%rsi)
	ret

L(less_128bytes):
	cmp	$64, %dl
	jb	L(less_64bytes)
	vmovups	%zmm2, (%rdi)
	vmovups	%zmm2, -0x40(%rsi)
	ret

L(less_64bytes):
	cmp	$32, %dl
	jb	L(less_32bytes)
	vmovdqu	%ymm2, (%rdi)
	vmovdqu %ymm2, -0x20(%rsi)
	ret

L(less_32bytes):
	vmovdqu %xmm0, (%rdi)
	vmovdqu %xmm0, -0x10(%rsi)
	ret

L(less_16bytes):
	cmp	$8, %dl
	jb	L(less_8bytes)
	vmovq	%xmm0, (%rdi)
	vmovq	%xmm0, -0x08(%rsi)
	ret

L(less_8bytes):
	vmovd	%xmm0, %ecx
	cmp	$4, %dl
	jb	L(less_4bytes)
	mov	%ecx, (%rdi)
	mov	%ecx, -0x04(%rsi)
	ret

L(less_4bytes):
	cmp	$2, %dl
	jb	L(less_2bytes)
	mov	%cx, (%rdi)
	mov	%cx, -0x02(%rsi)
	ret

L(less_2bytes):
	cmp	$1, %dl
	jb	L(less_1bytes)
	mov	%cl, (%rdi)
L(less_1bytes):
	ret

L(512bytesormore):
	mov	__x86_shared_cache_size_half(%rip), %rcx
	cmp	%rcx, %rdx
	ja	L(preloop_large)
	cmp	$1024, %rdx
	ja	L(1024bytesormore)

	vmovups	%zmm2, (%rdi)
	vmovups	%zmm2, 0x40(%rdi)
	vmovups	%zmm2, 0x80(%rdi)
	vmovups	%zmm2, 0xC0(%rdi)
	vmovups	%zmm2, 0x100(%rdi)
	vmovups	%zmm2, 0x140(%rdi)
	vmovups	%zmm2, 0x180(%rdi)
	vmovups	%zmm2, 0x1C0(%rdi)
	vmovups %zmm2, -0x200(%rsi)
	vmovups %zmm2, -0x1C0(%rsi)
	vmovups %zmm2, -0x180(%rsi)
	vmovups %zmm2, -0x140(%rsi)
	vmovups %zmm2, -0x100(%rsi)
	vmovups %zmm2, -0xC0(%rsi)
	vmovups %zmm2, -0x80(%rsi)
	vmovups %zmm2, -0x40(%rsi)
	ret

/* Align on 64 and loop with aligned stores.  */
L(1024bytesormore):
	sub	$0x100, %rsi
	vmovups	%zmm2, (%rax)
	and	$-0x40, %rdi
	add	$0x40, %rdi

L(gobble_256bytes_loop):
	vmovaps	%zmm2, (%rdi)
	vmovaps	%zmm2, 0x40(%rdi)
	vmovaps	%zmm2, 0x80(%rdi)
	vmovaps	%zmm2, 0xC0(%rdi)
	add	$0x100, %rdi
	cmp	%rsi, %rdi
	jb	L(gobble_256bytes_loop)
	vmovups %zmm2, (%rsi)
	vmovups %zmm2, 0x40(%rsi)
	vmovups %zmm2, 0x80(%rsi)
	vmovups %zmm2, 0xC0(%rsi)
	ret

/* Align on 128 and loop with non-temporal stores.  */
L(preloop_large):
	and	$-0x80, %rdi
	add	$0x80, %rdi
	vmovups	%zmm2, (%rax)
	vmovups	%zmm2, 0x40(%rax)
	sub	$0x200, %rsi

L(gobble_512bytes_nt_loop):
	vmovntdq %zmm2, (%rdi)
	vmovntdq %zmm2, 0x40(%rdi)
	vmovntdq %zmm2, 0x80(%rdi)
	vmovntdq %zmm2, 0xC0(%rdi)
	vmovntdq %zmm2, 0x100(%rdi)
	vmovntdq %zmm2, 0x140(%rdi)
	vmovntdq %zmm2, 0x180(%rdi)
	vmovntdq %zmm2, 0x1C0(%rdi)
	add	$0x200, %rdi
	cmp	%rsi, %rdi
	jb	L(gobble_512bytes_nt_loop)
	sfence
	vmovups %zmm2, (%rsi)
	vmovups %zmm2, 0x40(%rsi)
	vmovups %zmm2, 0x80(%rsi)
	vmovups %zmm2, 0xC0(%rsi)
	vmovups	%zmm2, 0x100(%rsi)
	vmovups	%zmm2, 0x140(%rsi)
	vmovups	%zmm2, 0x180(%rsi)
	vmovups	%zmm2, 0x1C0(%rsi)
	ret
END (MEMSET)
#endif

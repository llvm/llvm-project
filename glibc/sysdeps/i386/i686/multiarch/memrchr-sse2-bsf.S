/* Optimized memrchr with sse2
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
   Contributed by Intel Corporation.
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

#if IS_IN (libc)

# include <sysdep.h>

# define CFI_PUSH(REG)	\
	cfi_adjust_cfa_offset (4);	\
	cfi_rel_offset (REG, 0)

# define CFI_POP(REG)	\
	cfi_adjust_cfa_offset (-4);	\
	cfi_restore (REG)

# define PUSH(REG) pushl REG; CFI_PUSH (REG)
# define POP(REG) popl REG; CFI_POP (REG)

# define PARMS  4
# define STR1  PARMS
# define STR2  STR1+4
# define LEN   STR2+4

# define MEMCHR __memrchr_sse2_bsf

	.text
ENTRY (MEMCHR)
	mov	STR1(%esp), %ecx
	movd	STR2(%esp), %xmm1
	mov	LEN(%esp), %edx

	sub	$16, %edx
	jbe	L(length_less16)

	punpcklbw %xmm1, %xmm1
	add	%edx, %ecx
	punpcklbw %xmm1, %xmm1

	movdqu	(%ecx), %xmm0
	pshufd	$0, %xmm1, %xmm1
	pcmpeqb	%xmm1, %xmm0

/* Check if there is a match.  */
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches0)

	sub	$64, %ecx
	mov	%ecx, %eax
	and	$15, %eax
	jz	L(loop_prolog)

	add	$16, %ecx
	add	$16, %edx
	sub	%eax, %ecx
	sub	%eax, %edx

	.p2align 4
/* Loop start on aligned string.  */
L(loop_prolog):
	sub	$64, %edx
	jbe	L(exit_loop)

	movdqa	48(%ecx), %xmm0
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches48)

	movdqa	32(%ecx), %xmm2
	pcmpeqb	%xmm1, %xmm2
	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(matches32)

	movdqa	16(%ecx), %xmm3
	pcmpeqb	%xmm1, %xmm3
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches16)

	movdqa	(%ecx), %xmm4
	pcmpeqb	%xmm1, %xmm4
	pmovmskb %xmm4, %eax
	test	%eax, %eax
	jnz	L(matches0)

	sub	$64, %ecx
	sub	$64, %edx
	jbe	L(exit_loop)

	movdqa	48(%ecx), %xmm0
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches48)

	movdqa	32(%ecx), %xmm2
	pcmpeqb	%xmm1, %xmm2
	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(matches32)

	movdqa	16(%ecx), %xmm3
	pcmpeqb	%xmm1, %xmm3
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches16)

	movdqa	(%ecx), %xmm3
	pcmpeqb	%xmm1, %xmm3
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches0)

	mov	%ecx, %eax
	and	$63, %eax
	test	%eax, %eax
	jz	L(align64_loop)

	add	$64, %ecx
	add	$64, %edx
	sub	%eax, %ecx
	sub	%eax, %edx

	.p2align 4
L(align64_loop):
	sub	$64, %ecx
	sub	$64, %edx
	jbe	L(exit_loop)

	movdqa	(%ecx), %xmm0
	movdqa	16(%ecx), %xmm2
	movdqa	32(%ecx), %xmm3
	movdqa	48(%ecx), %xmm4

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm1, %xmm2
	pcmpeqb	%xmm1, %xmm3
	pcmpeqb	%xmm1, %xmm4

	pmaxub	%xmm3, %xmm0
	pmaxub	%xmm4, %xmm2
	pmaxub	%xmm0, %xmm2
	pmovmskb %xmm2, %eax

	test	%eax, %eax
	jz	L(align64_loop)

	pmovmskb %xmm4, %eax
	test	%eax, %eax
	jnz	L(matches48)

	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches32)

	movdqa	16(%ecx), %xmm2

	pcmpeqb	%xmm1, %xmm2
	pcmpeqb	(%ecx), %xmm1

	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(matches16)

	pmovmskb %xmm1, %eax
	bsr	%eax, %eax

	add	%ecx, %eax
	ret

	.p2align 4
L(exit_loop):
	add	$64, %edx
	cmp	$32, %edx
	jbe	L(exit_loop_32)

	movdqa	48(%ecx), %xmm0
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches48)

	movdqa	32(%ecx), %xmm2
	pcmpeqb	%xmm1, %xmm2
	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(matches32)

	movdqa	16(%ecx), %xmm3
	pcmpeqb	%xmm1, %xmm3
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches16_1)
	cmp	$48, %edx
	jbe	L(return_null)

	pcmpeqb	(%ecx), %xmm1
	pmovmskb %xmm1, %eax
	test	%eax, %eax
	jnz	L(matches0_1)
	xor	%eax, %eax
	ret

	.p2align 4
L(exit_loop_32):
	movdqa	48(%ecx), %xmm0
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches48_1)
	cmp	$16, %edx
	jbe	L(return_null)

	pcmpeqb	32(%ecx), %xmm1
	pmovmskb %xmm1, %eax
	test	%eax, %eax
	jnz	L(matches32_1)
	xor	%eax, %eax
	ret

	.p2align 4
L(matches0):
	bsr	%eax, %eax
	add	%ecx, %eax
	ret

	.p2align 4
L(matches16):
	bsr	%eax, %eax
	lea	16(%eax, %ecx), %eax
	ret

	.p2align 4
L(matches32):
	bsr	%eax, %eax
	lea	32(%eax, %ecx), %eax
	ret

	.p2align 4
L(matches48):
	bsr	%eax, %eax
	lea	48(%eax, %ecx), %eax
	ret

	.p2align 4
L(matches0_1):
	bsr	%eax, %eax
	sub	$64, %edx
	add	%eax, %edx
	jl	L(return_null)
	add	%ecx, %eax
	ret

	.p2align 4
L(matches16_1):
	bsr	%eax, %eax
	sub	$48, %edx
	add	%eax, %edx
	jl	L(return_null)
	lea	16(%ecx, %eax), %eax
	ret

	.p2align 4
L(matches32_1):
	bsr	%eax, %eax
	sub	$32, %edx
	add	%eax, %edx
	jl	L(return_null)
	lea	32(%ecx, %eax), %eax
	ret

	.p2align 4
L(matches48_1):
	bsr	%eax, %eax
	sub	$16, %edx
	add	%eax, %edx
	jl	L(return_null)
	lea	48(%ecx, %eax), %eax
	ret

	.p2align 4
L(return_null):
	xor	%eax, %eax
	ret

	.p2align 4
L(length_less16_offset0):
	mov	%dl, %cl
	pcmpeqb	(%eax), %xmm1

	mov	$1, %edx
	sal	%cl, %edx
	sub	$1, %edx
	mov	%edx, %ecx

	pmovmskb %xmm1, %edx

	and	%ecx, %edx
	test	%edx, %edx
	jz	L(return_null)

	bsr	%edx, %ecx
	add	%ecx, %eax
	ret

	.p2align 4
L(length_less16):
	punpcklbw %xmm1, %xmm1
	mov	%ecx, %eax
	punpcklbw %xmm1, %xmm1
	add	$16, %edx
	jz	L(return_null)

	pshufd	$0, %xmm1, %xmm1
	and	$15, %ecx
	jz	L(length_less16_offset0)

	PUSH	(%edi)
	mov	%cl, %dh
	add	%dl, %dh
	and	$-16, %eax

	sub	$16, %dh
	ja	L(length_less16_part2)

	pcmpeqb	(%eax), %xmm1
	pmovmskb %xmm1, %edi

	sar	%cl, %edi
	add	%ecx, %eax
	mov	%dl, %cl

	mov	$1, %edx
	sal	%cl, %edx
	sub	$1, %edx

	and	%edx, %edi
	test	%edi, %edi
	jz	L(ret_null)

	bsr	%edi, %edi
	add	%edi, %eax
	POP	(%edi)
	ret

	CFI_PUSH     (%edi)

	.p2align 4
L(length_less16_part2):
	movdqa	16(%eax), %xmm2
	pcmpeqb	%xmm1, %xmm2
	pmovmskb %xmm2, %edi

	mov	%cl, %ch

	mov	%dh, %cl
	mov	$1, %edx
	sal	%cl, %edx
	sub	$1, %edx

	and	%edx, %edi

	test	%edi, %edi
	jnz	L(length_less16_part2_return)

	pcmpeqb	(%eax), %xmm1
	pmovmskb %xmm1, %edi

	mov	%ch, %cl
	sar	%cl, %edi
	test	%edi, %edi
	jz	L(ret_null)

	bsr	%edi, %edi
	add	%edi, %eax
	xor	%ch, %ch
	add	%ecx, %eax
	POP	(%edi)
	ret

	CFI_PUSH     (%edi)

	.p2align 4
L(length_less16_part2_return):
	bsr	%edi, %edi
	lea	16(%eax, %edi), %eax
	POP	(%edi)
	ret

	CFI_PUSH     (%edi)

	.p2align 4
L(ret_null):
	xor	%eax, %eax
	POP	(%edi)
	ret

END (MEMCHR)
#endif

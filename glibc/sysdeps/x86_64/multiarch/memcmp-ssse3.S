/* memcmp with SSSE3, wmemcmp with SSSE3
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

# ifndef MEMCMP
#  define MEMCMP	__memcmp_ssse3
# endif

/* Warning!
	   wmemcmp has to use SIGNED comparison for elements.
	   memcmp has to use UNSIGNED comparison for elemnts.
*/

	atom_text_section
ENTRY (MEMCMP)
# ifdef USE_AS_WMEMCMP
	shl	$2, %RDX_LP
	test	%RDX_LP, %RDX_LP
	jz	L(equal)
# elif defined __ILP32__
	/* Clear the upper 32 bits.  */
	mov	%edx, %edx
# endif
	mov	%rdx, %rcx
	mov	%rdi, %rdx
	cmp	$48, %rcx;
	jae	L(48bytesormore)	/* LEN => 48  */

	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
/* ECX >= 32.  */
L(48bytesormore):
	movdqu	(%rdi), %xmm3
	movdqu	(%rsi), %xmm0
	pcmpeqb	%xmm0, %xmm3
	pmovmskb %xmm3, %edx
	lea	16(%rdi), %rdi
	lea	16(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(less16bytes)
	mov	%edi, %edx
	and	$0xf, %edx
	xor	%rdx, %rdi
	sub	%rdx, %rsi
	add	%rdx, %rcx
	mov	%esi, %edx
	and	$0xf, %edx
	jz	L(shr_0)
	xor	%rdx, %rsi

# ifndef USE_AS_WMEMCMP
	cmp	$8, %edx
	jae	L(next_unaligned_table)
	cmp	$0, %edx
	je	L(shr_0)
	cmp	$1, %edx
	je	L(shr_1)
	cmp	$2, %edx
	je	L(shr_2)
	cmp	$3, %edx
	je	L(shr_3)
	cmp	$4, %edx
	je	L(shr_4)
	cmp	$5, %edx
	je	L(shr_5)
	cmp	$6, %edx
	je	L(shr_6)
	jmp	L(shr_7)

	.p2align 2
L(next_unaligned_table):
	cmp	$8, %edx
	je	L(shr_8)
	cmp	$9, %edx
	je	L(shr_9)
	cmp	$10, %edx
	je	L(shr_10)
	cmp	$11, %edx
	je	L(shr_11)
	cmp	$12, %edx
	je	L(shr_12)
	cmp	$13, %edx
	je	L(shr_13)
	cmp	$14, %edx
	je	L(shr_14)
	jmp	L(shr_15)
# else
	cmp	$0, %edx
	je	L(shr_0)
	cmp	$4, %edx
	je	L(shr_4)
	cmp	$8, %edx
	je	L(shr_8)
	jmp	L(shr_12)
# endif

	.p2align 4
L(shr_0):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	jae	L(shr_0_gobble)
	xor	%eax, %eax
	movdqa	(%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1
	movdqa	16(%rsi), %xmm2
	pcmpeqb	16(%rdi), %xmm2
	pand	%xmm1, %xmm2
	pmovmskb %xmm2, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_0_gobble):
	movdqa	(%rsi), %xmm0
	xor	%eax, %eax
	pcmpeqb	(%rdi), %xmm0
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm2
	pcmpeqb	16(%rdi), %xmm2
L(shr_0_gobble_loop):
	pand	%xmm0, %xmm2
	sub	$32, %rcx
	pmovmskb %xmm2, %edx
	movdqa	%xmm0, %xmm1
	movdqa	32(%rsi), %xmm0
	movdqa	48(%rsi), %xmm2
	sbb	$0xffff, %edx
	pcmpeqb	32(%rdi), %xmm0
	pcmpeqb	48(%rdi), %xmm2
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	jz	L(shr_0_gobble_loop)

	pand	%xmm0, %xmm2
	cmp	$0, %rcx
	jge	L(next)
	inc	%edx
	add	$32, %rcx
L(next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm2, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

# ifndef USE_AS_WMEMCMP

	.p2align 4
L(shr_1):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	mov	%edx, %eax
	jae	L(shr_1_gobble)

	movdqa	16(%rsi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$1, (%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1

	movdqa	32(%rsi), %xmm3
	palignr	$1, %xmm2, %xmm3
	pcmpeqb	16(%rdi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	$1, %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_1_gobble):
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm0
	palignr	$1, (%rsi), %xmm0
	pcmpeqb	(%rdi), %xmm0

	movdqa	32(%rsi), %xmm3
	palignr	$1, 16(%rsi), %xmm3
	pcmpeqb	16(%rdi), %xmm3

L(shr_1_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %rcx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%rsi), %xmm3
	palignr	$1, 48(%rsi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%rsi), %xmm0
	palignr	$1, 32(%rsi), %xmm0
	pcmpeqb	32(%rdi), %xmm0
	lea	32(%rsi), %rsi
	pcmpeqb	48(%rdi), %xmm3

	lea	32(%rdi), %rdi
	jz	L(shr_1_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %rcx
	jge	L(shr_1_gobble_next)
	inc	%edx
	add	$32, %rcx
L(shr_1_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	1(%rsi), %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)


	.p2align 4
L(shr_2):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	mov	%edx, %eax
	jae	L(shr_2_gobble)

	movdqa	16(%rsi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$2, (%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1

	movdqa	32(%rsi), %xmm3
	palignr	$2, %xmm2, %xmm3
	pcmpeqb	16(%rdi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	$2, %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_2_gobble):
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm0
	palignr	$2, (%rsi), %xmm0
	pcmpeqb	(%rdi), %xmm0

	movdqa	32(%rsi), %xmm3
	palignr	$2, 16(%rsi), %xmm3
	pcmpeqb	16(%rdi), %xmm3

L(shr_2_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %rcx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%rsi), %xmm3
	palignr	$2, 48(%rsi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%rsi), %xmm0
	palignr	$2, 32(%rsi), %xmm0
	pcmpeqb	32(%rdi), %xmm0
	lea	32(%rsi), %rsi
	pcmpeqb	48(%rdi), %xmm3

	lea	32(%rdi), %rdi
	jz	L(shr_2_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %rcx
	jge	L(shr_2_gobble_next)
	inc	%edx
	add	$32, %rcx
L(shr_2_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	2(%rsi), %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_3):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	mov	%edx, %eax
	jae	L(shr_3_gobble)

	movdqa	16(%rsi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$3, (%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1

	movdqa	32(%rsi), %xmm3
	palignr	$3, %xmm2, %xmm3
	pcmpeqb	16(%rdi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	$3, %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_3_gobble):
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm0
	palignr	$3, (%rsi), %xmm0
	pcmpeqb	(%rdi), %xmm0

	movdqa	32(%rsi), %xmm3
	palignr	$3, 16(%rsi), %xmm3
	pcmpeqb	16(%rdi), %xmm3

L(shr_3_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %rcx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%rsi), %xmm3
	palignr	$3, 48(%rsi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%rsi), %xmm0
	palignr	$3, 32(%rsi), %xmm0
	pcmpeqb	32(%rdi), %xmm0
	lea	32(%rsi), %rsi
	pcmpeqb	48(%rdi), %xmm3

	lea	32(%rdi), %rdi
	jz	L(shr_3_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %rcx
	jge	L(shr_3_gobble_next)
	inc	%edx
	add	$32, %rcx
L(shr_3_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	3(%rsi), %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

# endif

	.p2align 4
L(shr_4):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	mov	%edx, %eax
	jae	L(shr_4_gobble)

	movdqa	16(%rsi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$4, (%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1

	movdqa	32(%rsi), %xmm3
	palignr	$4, %xmm2, %xmm3
	pcmpeqb	16(%rdi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	$4, %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_4_gobble):
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm0
	palignr	$4, (%rsi), %xmm0
	pcmpeqb	(%rdi), %xmm0

	movdqa	32(%rsi), %xmm3
	palignr	$4, 16(%rsi), %xmm3
	pcmpeqb	16(%rdi), %xmm3

L(shr_4_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %rcx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%rsi), %xmm3
	palignr	$4, 48(%rsi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%rsi), %xmm0
	palignr	$4, 32(%rsi), %xmm0
	pcmpeqb	32(%rdi), %xmm0
	lea	32(%rsi), %rsi
	pcmpeqb	48(%rdi), %xmm3

	lea	32(%rdi), %rdi
	jz	L(shr_4_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %rcx
	jge	L(shr_4_gobble_next)
	inc	%edx
	add	$32, %rcx
L(shr_4_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	4(%rsi), %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

# ifndef USE_AS_WMEMCMP

	.p2align 4
L(shr_5):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	mov	%edx, %eax
	jae	L(shr_5_gobble)

	movdqa	16(%rsi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$5, (%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1

	movdqa	32(%rsi), %xmm3
	palignr	$5, %xmm2, %xmm3
	pcmpeqb	16(%rdi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	$5, %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_5_gobble):
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm0
	palignr	$5, (%rsi), %xmm0
	pcmpeqb	(%rdi), %xmm0

	movdqa	32(%rsi), %xmm3
	palignr	$5, 16(%rsi), %xmm3
	pcmpeqb	16(%rdi), %xmm3

L(shr_5_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %rcx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%rsi), %xmm3
	palignr	$5, 48(%rsi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%rsi), %xmm0
	palignr	$5, 32(%rsi), %xmm0
	pcmpeqb	32(%rdi), %xmm0
	lea	32(%rsi), %rsi
	pcmpeqb	48(%rdi), %xmm3

	lea	32(%rdi), %rdi
	jz	L(shr_5_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %rcx
	jge	L(shr_5_gobble_next)
	inc	%edx
	add	$32, %rcx
L(shr_5_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	5(%rsi), %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_6):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	mov	%edx, %eax
	jae	L(shr_6_gobble)

	movdqa	16(%rsi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$6, (%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1

	movdqa	32(%rsi), %xmm3
	palignr	$6, %xmm2, %xmm3
	pcmpeqb	16(%rdi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	$6, %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_6_gobble):
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm0
	palignr	$6, (%rsi), %xmm0
	pcmpeqb	(%rdi), %xmm0

	movdqa	32(%rsi), %xmm3
	palignr	$6, 16(%rsi), %xmm3
	pcmpeqb	16(%rdi), %xmm3

L(shr_6_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %rcx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%rsi), %xmm3
	palignr	$6, 48(%rsi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%rsi), %xmm0
	palignr	$6, 32(%rsi), %xmm0
	pcmpeqb	32(%rdi), %xmm0
	lea	32(%rsi), %rsi
	pcmpeqb	48(%rdi), %xmm3

	lea	32(%rdi), %rdi
	jz	L(shr_6_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %rcx
	jge	L(shr_6_gobble_next)
	inc	%edx
	add	$32, %rcx
L(shr_6_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	6(%rsi), %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_7):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	mov	%edx, %eax
	jae	L(shr_7_gobble)

	movdqa	16(%rsi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$7, (%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1

	movdqa	32(%rsi), %xmm3
	palignr	$7, %xmm2, %xmm3
	pcmpeqb	16(%rdi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	$7, %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_7_gobble):
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm0
	palignr	$7, (%rsi), %xmm0
	pcmpeqb	(%rdi), %xmm0

	movdqa	32(%rsi), %xmm3
	palignr	$7, 16(%rsi), %xmm3
	pcmpeqb	16(%rdi), %xmm3

L(shr_7_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %rcx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%rsi), %xmm3
	palignr	$7, 48(%rsi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%rsi), %xmm0
	palignr	$7, 32(%rsi), %xmm0
	pcmpeqb	32(%rdi), %xmm0
	lea	32(%rsi), %rsi
	pcmpeqb	48(%rdi), %xmm3

	lea	32(%rdi), %rdi
	jz	L(shr_7_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %rcx
	jge	L(shr_7_gobble_next)
	inc	%edx
	add	$32, %rcx
L(shr_7_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	7(%rsi), %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

# endif

	.p2align 4
L(shr_8):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	mov	%edx, %eax
	jae	L(shr_8_gobble)

	movdqa	16(%rsi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$8, (%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1

	movdqa	32(%rsi), %xmm3
	palignr	$8, %xmm2, %xmm3
	pcmpeqb	16(%rdi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	$8, %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_8_gobble):
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm0
	palignr	$8, (%rsi), %xmm0
	pcmpeqb	(%rdi), %xmm0

	movdqa	32(%rsi), %xmm3
	palignr	$8, 16(%rsi), %xmm3
	pcmpeqb	16(%rdi), %xmm3

L(shr_8_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %rcx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%rsi), %xmm3
	palignr	$8, 48(%rsi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%rsi), %xmm0
	palignr	$8, 32(%rsi), %xmm0
	pcmpeqb	32(%rdi), %xmm0
	lea	32(%rsi), %rsi
	pcmpeqb	48(%rdi), %xmm3

	lea	32(%rdi), %rdi
	jz	L(shr_8_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %rcx
	jge	L(shr_8_gobble_next)
	inc	%edx
	add	$32, %rcx
L(shr_8_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	8(%rsi), %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

# ifndef USE_AS_WMEMCMP

	.p2align 4
L(shr_9):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	mov	%edx, %eax
	jae	L(shr_9_gobble)

	movdqa	16(%rsi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$9, (%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1

	movdqa	32(%rsi), %xmm3
	palignr	$9, %xmm2, %xmm3
	pcmpeqb	16(%rdi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	$9, %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_9_gobble):
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm0
	palignr	$9, (%rsi), %xmm0
	pcmpeqb	(%rdi), %xmm0

	movdqa	32(%rsi), %xmm3
	palignr	$9, 16(%rsi), %xmm3
	pcmpeqb	16(%rdi), %xmm3

L(shr_9_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %rcx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%rsi), %xmm3
	palignr	$9, 48(%rsi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%rsi), %xmm0
	palignr	$9, 32(%rsi), %xmm0
	pcmpeqb	32(%rdi), %xmm0
	lea	32(%rsi), %rsi
	pcmpeqb	48(%rdi), %xmm3

	lea	32(%rdi), %rdi
	jz	L(shr_9_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %rcx
	jge	L(shr_9_gobble_next)
	inc	%edx
	add	$32, %rcx
L(shr_9_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	9(%rsi), %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_10):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	mov	%edx, %eax
	jae	L(shr_10_gobble)

	movdqa	16(%rsi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$10, (%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1

	movdqa	32(%rsi), %xmm3
	palignr	$10, %xmm2, %xmm3
	pcmpeqb	16(%rdi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	$10, %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_10_gobble):
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm0
	palignr	$10, (%rsi), %xmm0
	pcmpeqb	(%rdi), %xmm0

	movdqa	32(%rsi), %xmm3
	palignr	$10, 16(%rsi), %xmm3
	pcmpeqb	16(%rdi), %xmm3

L(shr_10_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %rcx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%rsi), %xmm3
	palignr	$10, 48(%rsi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%rsi), %xmm0
	palignr	$10, 32(%rsi), %xmm0
	pcmpeqb	32(%rdi), %xmm0
	lea	32(%rsi), %rsi
	pcmpeqb	48(%rdi), %xmm3

	lea	32(%rdi), %rdi
	jz	L(shr_10_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %rcx
	jge	L(shr_10_gobble_next)
	inc	%edx
	add	$32, %rcx
L(shr_10_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	10(%rsi), %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_11):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	mov	%edx, %eax
	jae	L(shr_11_gobble)

	movdqa	16(%rsi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$11, (%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1

	movdqa	32(%rsi), %xmm3
	palignr	$11, %xmm2, %xmm3
	pcmpeqb	16(%rdi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	$11, %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_11_gobble):
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm0
	palignr	$11, (%rsi), %xmm0
	pcmpeqb	(%rdi), %xmm0

	movdqa	32(%rsi), %xmm3
	palignr	$11, 16(%rsi), %xmm3
	pcmpeqb	16(%rdi), %xmm3

L(shr_11_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %rcx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%rsi), %xmm3
	palignr	$11, 48(%rsi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%rsi), %xmm0
	palignr	$11, 32(%rsi), %xmm0
	pcmpeqb	32(%rdi), %xmm0
	lea	32(%rsi), %rsi
	pcmpeqb	48(%rdi), %xmm3

	lea	32(%rdi), %rdi
	jz	L(shr_11_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %rcx
	jge	L(shr_11_gobble_next)
	inc	%edx
	add	$32, %rcx
L(shr_11_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	11(%rsi), %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

# endif

	.p2align 4
L(shr_12):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	mov	%edx, %eax
	jae	L(shr_12_gobble)

	movdqa	16(%rsi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$12, (%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1

	movdqa	32(%rsi), %xmm3
	palignr	$12, %xmm2, %xmm3
	pcmpeqb	16(%rdi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	$12, %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_12_gobble):
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm0
	palignr	$12, (%rsi), %xmm0
	pcmpeqb	(%rdi), %xmm0

	movdqa	32(%rsi), %xmm3
	palignr	$12, 16(%rsi), %xmm3
	pcmpeqb	16(%rdi), %xmm3

L(shr_12_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %rcx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%rsi), %xmm3
	palignr	$12, 48(%rsi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%rsi), %xmm0
	palignr	$12, 32(%rsi), %xmm0
	pcmpeqb	32(%rdi), %xmm0
	lea	32(%rsi), %rsi
	pcmpeqb	48(%rdi), %xmm3

	lea	32(%rdi), %rdi
	jz	L(shr_12_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %rcx
	jge	L(shr_12_gobble_next)
	inc	%edx
	add	$32, %rcx
L(shr_12_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	12(%rsi), %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

# ifndef USE_AS_WMEMCMP

	.p2align 4
L(shr_13):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	mov	%edx, %eax
	jae	L(shr_13_gobble)

	movdqa	16(%rsi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$13, (%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1

	movdqa	32(%rsi), %xmm3
	palignr	$13, %xmm2, %xmm3
	pcmpeqb	16(%rdi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	$13, %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_13_gobble):
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm0
	palignr	$13, (%rsi), %xmm0
	pcmpeqb	(%rdi), %xmm0

	movdqa	32(%rsi), %xmm3
	palignr	$13, 16(%rsi), %xmm3
	pcmpeqb	16(%rdi), %xmm3

L(shr_13_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %rcx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%rsi), %xmm3
	palignr	$13, 48(%rsi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%rsi), %xmm0
	palignr	$13, 32(%rsi), %xmm0
	pcmpeqb	32(%rdi), %xmm0
	lea	32(%rsi), %rsi
	pcmpeqb	48(%rdi), %xmm3

	lea	32(%rdi), %rdi
	jz	L(shr_13_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %rcx
	jge	L(shr_13_gobble_next)
	inc	%edx
	add	$32, %rcx
L(shr_13_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	13(%rsi), %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_14):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	mov	%edx, %eax
	jae	L(shr_14_gobble)

	movdqa	16(%rsi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$14, (%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1

	movdqa	32(%rsi), %xmm3
	palignr	$14, %xmm2, %xmm3
	pcmpeqb	16(%rdi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	$14, %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_14_gobble):
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm0
	palignr	$14, (%rsi), %xmm0
	pcmpeqb	(%rdi), %xmm0

	movdqa	32(%rsi), %xmm3
	palignr	$14, 16(%rsi), %xmm3
	pcmpeqb	16(%rdi), %xmm3

L(shr_14_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %rcx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%rsi), %xmm3
	palignr	$14, 48(%rsi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%rsi), %xmm0
	palignr	$14, 32(%rsi), %xmm0
	pcmpeqb	32(%rdi), %xmm0
	lea	32(%rsi), %rsi
	pcmpeqb	48(%rdi), %xmm3

	lea	32(%rdi), %rdi
	jz	L(shr_14_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %rcx
	jge	L(shr_14_gobble_next)
	inc	%edx
	add	$32, %rcx
L(shr_14_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	14(%rsi), %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_15):
	cmp	$80, %rcx
	lea	-48(%rcx), %rcx
	mov	%edx, %eax
	jae	L(shr_15_gobble)

	movdqa	16(%rsi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$15, (%rsi), %xmm1
	pcmpeqb	(%rdi), %xmm1

	movdqa	32(%rsi), %xmm3
	palignr	$15, %xmm2, %xmm3
	pcmpeqb	16(%rdi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)
	add	$15, %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)

	.p2align 4
L(shr_15_gobble):
	sub	$32, %rcx
	movdqa	16(%rsi), %xmm0
	palignr	$15, (%rsi), %xmm0
	pcmpeqb	(%rdi), %xmm0

	movdqa	32(%rsi), %xmm3
	palignr	$15, 16(%rsi), %xmm3
	pcmpeqb	16(%rdi), %xmm3

L(shr_15_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %rcx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%rsi), %xmm3
	palignr	$15, 48(%rsi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%rsi), %xmm0
	palignr	$15, 32(%rsi), %xmm0
	pcmpeqb	32(%rdi), %xmm0
	lea	32(%rsi), %rsi
	pcmpeqb	48(%rdi), %xmm3

	lea	32(%rdi), %rdi
	jz	L(shr_15_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %rcx
	jge	L(shr_15_gobble_next)
	inc	%edx
	add	$32, %rcx
L(shr_15_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%rdi), %rdi
	lea	32(%rsi), %rsi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	15(%rsi), %rsi
	add	%rcx, %rsi
	add	%rcx, %rdi
	jmp	L(less48bytes)
# endif
	.p2align 4
L(exit):
	pmovmskb %xmm1, %r8d
	sub	$0xffff, %r8d
	jz	L(first16bytes)
	lea	-16(%rsi), %rsi
	lea	-16(%rdi), %rdi
	mov	%r8d, %edx
L(first16bytes):
	add	%rax, %rsi
L(less16bytes):
# ifndef USE_AS_WMEMCMP
	test	%dl, %dl
	jz	L(next_24_bytes)

	test	$0x01, %dl
	jnz	L(Byte16)

	test	$0x02, %dl
	jnz	L(Byte17)

	test	$0x04, %dl
	jnz	L(Byte18)

	test	$0x08, %dl
	jnz	L(Byte19)

	test	$0x10, %dl
	jnz	L(Byte20)

	test	$0x20, %dl
	jnz	L(Byte21)

	test	$0x40, %dl
	jnz	L(Byte22)

	movzbl	-9(%rdi), %eax
	movzbl	-9(%rsi), %edx
	sub	%edx, %eax
	ret

	.p2align 4
L(Byte16):
	movzbl	-16(%rdi), %eax
	movzbl	-16(%rsi), %edx
	sub	%edx, %eax
	ret

	.p2align 4
L(Byte17):
	movzbl	-15(%rdi), %eax
	movzbl	-15(%rsi), %edx
	sub	%edx, %eax
	ret

	.p2align 4
L(Byte18):
	movzbl	-14(%rdi), %eax
	movzbl	-14(%rsi), %edx
	sub	%edx, %eax
	ret

	.p2align 4
L(Byte19):
	movzbl	-13(%rdi), %eax
	movzbl	-13(%rsi), %edx
	sub	%edx, %eax
	ret

	.p2align 4
L(Byte20):
	movzbl	-12(%rdi), %eax
	movzbl	-12(%rsi), %edx
	sub	%edx, %eax
	ret

	.p2align 4
L(Byte21):
	movzbl	-11(%rdi), %eax
	movzbl	-11(%rsi), %edx
	sub	%edx, %eax
	ret

	.p2align 4
L(Byte22):
	movzbl	-10(%rdi), %eax
	movzbl	-10(%rsi), %edx
	sub	%edx, %eax
	ret

	.p2align 4
L(next_24_bytes):
	lea	8(%rdi), %rdi
	lea	8(%rsi), %rsi
	test	$0x01, %dh
	jnz	L(Byte16)

	test	$0x02, %dh
	jnz	L(Byte17)

	test	$0x04, %dh
	jnz	L(Byte18)

	test	$0x08, %dh
	jnz	L(Byte19)

	test	$0x10, %dh
	jnz	L(Byte20)

	test	$0x20, %dh
	jnz	L(Byte21)

	test	$0x40, %dh
	jnz	L(Byte22)

	movzbl	-9(%rdi), %eax
	movzbl	-9(%rsi), %edx
	sub	%edx, %eax
	ret
# else
/* special for wmemcmp */
	xor	%eax, %eax
	test	%dl, %dl
	jz	L(next_two_double_words)
	and	$15, %dl
	jz	L(second_double_word)
	mov	-16(%rdi), %eax
	cmp	-16(%rsi), %eax
	jne	L(find_diff)
	ret

	.p2align 4
L(second_double_word):
	mov	-12(%rdi), %eax
	cmp	-12(%rsi), %eax
	jne	L(find_diff)
	ret

	.p2align 4
L(next_two_double_words):
	and	$15, %dh
	jz	L(fourth_double_word)
	mov	-8(%rdi), %eax
	cmp	-8(%rsi), %eax
	jne	L(find_diff)
	ret

	.p2align 4
L(fourth_double_word):
	mov	-4(%rdi), %eax
	cmp	-4(%rsi), %eax
	jne	L(find_diff)
	ret
# endif

	.p2align 4
L(less48bytes):
	cmp	$8, %ecx
	jae	L(more8bytes)
	cmp	$0, %ecx
	je	L(0bytes)
# ifndef USE_AS_WMEMCMP
	cmp	$1, %ecx
	je	L(1bytes)
	cmp	$2, %ecx
	je	L(2bytes)
	cmp	$3, %ecx
	je	L(3bytes)
	cmp	$4, %ecx
	je	L(4bytes)
	cmp	$5, %ecx
	je	L(5bytes)
	cmp	$6, %ecx
	je	L(6bytes)
	jmp	L(7bytes)
# else
	jmp	L(4bytes)
# endif

	.p2align 4
L(more8bytes):
	cmp	$16, %ecx
	jae	L(more16bytes)
	cmp	$8, %ecx
	je	L(8bytes)
# ifndef USE_AS_WMEMCMP
	cmp	$9, %ecx
	je	L(9bytes)
	cmp	$10, %ecx
	je	L(10bytes)
	cmp	$11, %ecx
	je	L(11bytes)
	cmp	$12, %ecx
	je	L(12bytes)
	cmp	$13, %ecx
	je	L(13bytes)
	cmp	$14, %ecx
	je	L(14bytes)
	jmp	L(15bytes)
# else
	jmp	L(12bytes)
# endif

	.p2align 4
L(more16bytes):
	cmp	$24, %ecx
	jae	L(more24bytes)
	cmp	$16, %ecx
	je	L(16bytes)
# ifndef USE_AS_WMEMCMP
	cmp	$17, %ecx
	je	L(17bytes)
	cmp	$18, %ecx
	je	L(18bytes)
	cmp	$19, %ecx
	je	L(19bytes)
	cmp	$20, %ecx
	je	L(20bytes)
	cmp	$21, %ecx
	je	L(21bytes)
	cmp	$22, %ecx
	je	L(22bytes)
	jmp	L(23bytes)
# else
	jmp	L(20bytes)
# endif

	.p2align 4
L(more24bytes):
	cmp	$32, %ecx
	jae	L(more32bytes)
	cmp	$24, %ecx
	je	L(24bytes)
# ifndef USE_AS_WMEMCMP
	cmp	$25, %ecx
	je	L(25bytes)
	cmp	$26, %ecx
	je	L(26bytes)
	cmp	$27, %ecx
	je	L(27bytes)
	cmp	$28, %ecx
	je	L(28bytes)
	cmp	$29, %ecx
	je	L(29bytes)
	cmp	$30, %ecx
	je	L(30bytes)
	jmp	L(31bytes)
# else
	jmp	L(28bytes)
# endif

	.p2align 4
L(more32bytes):
	cmp	$40, %ecx
	jae	L(more40bytes)
	cmp	$32, %ecx
	je	L(32bytes)
# ifndef USE_AS_WMEMCMP
	cmp	$33, %ecx
	je	L(33bytes)
	cmp	$34, %ecx
	je	L(34bytes)
	cmp	$35, %ecx
	je	L(35bytes)
	cmp	$36, %ecx
	je	L(36bytes)
	cmp	$37, %ecx
	je	L(37bytes)
	cmp	$38, %ecx
	je	L(38bytes)
	jmp	L(39bytes)
# else
	jmp	L(36bytes)
# endif

	.p2align 4
L(more40bytes):
	cmp	$40, %ecx
	je	L(40bytes)
# ifndef USE_AS_WMEMCMP
	cmp	$41, %ecx
	je	L(41bytes)
	cmp	$42, %ecx
	je	L(42bytes)
	cmp	$43, %ecx
	je	L(43bytes)
	cmp	$44, %ecx
	je	L(44bytes)
	cmp	$45, %ecx
	je	L(45bytes)
	cmp	$46, %ecx
	je	L(46bytes)
	jmp	L(47bytes)

	.p2align 4
L(44bytes):
	movl	-44(%rdi), %eax
	movl	-44(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(40bytes):
	movl	-40(%rdi), %eax
	movl	-40(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(36bytes):
	movl	-36(%rdi), %eax
	movl	-36(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(32bytes):
	movl	-32(%rdi), %eax
	movl	-32(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(28bytes):
	movl	-28(%rdi), %eax
	movl	-28(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(24bytes):
	movl	-24(%rdi), %eax
	movl	-24(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(20bytes):
	movl	-20(%rdi), %eax
	movl	-20(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(16bytes):
	movl	-16(%rdi), %eax
	movl	-16(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(12bytes):
	movl	-12(%rdi), %eax
	movl	-12(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(8bytes):
	movl	-8(%rdi), %eax
	movl	-8(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(4bytes):
	movl	-4(%rdi), %eax
	movl	-4(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(0bytes):
	xor	%eax, %eax
	ret
# else
	.p2align 4
L(44bytes):
	movl	-44(%rdi), %eax
	cmp	-44(%rsi), %eax
	jne	L(find_diff)
L(40bytes):
	movl	-40(%rdi), %eax
	cmp	-40(%rsi), %eax
	jne	L(find_diff)
L(36bytes):
	movl	-36(%rdi), %eax
	cmp	-36(%rsi), %eax
	jne	L(find_diff)
L(32bytes):
	movl	-32(%rdi), %eax
	cmp	-32(%rsi), %eax
	jne	L(find_diff)
L(28bytes):
	movl	-28(%rdi), %eax
	cmp	-28(%rsi), %eax
	jne	L(find_diff)
L(24bytes):
	movl	-24(%rdi), %eax
	cmp	-24(%rsi), %eax
	jne	L(find_diff)
L(20bytes):
	movl	-20(%rdi), %eax
	cmp	-20(%rsi), %eax
	jne	L(find_diff)
L(16bytes):
	movl	-16(%rdi), %eax
	cmp	-16(%rsi), %eax
	jne	L(find_diff)
L(12bytes):
	movl	-12(%rdi), %eax
	cmp	-12(%rsi), %eax
	jne	L(find_diff)
L(8bytes):
	movl	-8(%rdi), %eax
	cmp	-8(%rsi), %eax
	jne	L(find_diff)
L(4bytes):
	movl	-4(%rdi), %eax
	cmp	-4(%rsi), %eax
	jne	L(find_diff)
L(0bytes):
	xor	%eax, %eax
	ret
# endif

# ifndef USE_AS_WMEMCMP
	.p2align 4
L(45bytes):
	movl	-45(%rdi), %eax
	movl	-45(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(41bytes):
	movl	-41(%rdi), %eax
	movl	-41(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(37bytes):
	movl	-37(%rdi), %eax
	movl	-37(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(33bytes):
	movl	-33(%rdi), %eax
	movl	-33(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(29bytes):
	movl	-29(%rdi), %eax
	movl	-29(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(25bytes):
	movl	-25(%rdi), %eax
	movl	-25(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(21bytes):
	movl	-21(%rdi), %eax
	movl	-21(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(17bytes):
	movl	-17(%rdi), %eax
	movl	-17(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(13bytes):
	movl	-13(%rdi), %eax
	movl	-13(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(9bytes):
	movl	-9(%rdi), %eax
	movl	-9(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(5bytes):
	movl	-5(%rdi), %eax
	movl	-5(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(1bytes):
	movzbl	-1(%rdi), %eax
	cmpb	-1(%rsi), %al
	jne	L(set)
	xor	%eax, %eax
	ret

	.p2align 4
L(46bytes):
	movl	-46(%rdi), %eax
	movl	-46(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(42bytes):
	movl	-42(%rdi), %eax
	movl	-42(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(38bytes):
	movl	-38(%rdi), %eax
	movl	-38(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(34bytes):
	movl	-34(%rdi), %eax
	movl	-34(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(30bytes):
	movl	-30(%rdi), %eax
	movl	-30(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(26bytes):
	movl	-26(%rdi), %eax
	movl	-26(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(22bytes):
	movl	-22(%rdi), %eax
	movl	-22(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(18bytes):
	movl	-18(%rdi), %eax
	movl	-18(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(14bytes):
	movl	-14(%rdi), %eax
	movl	-14(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(10bytes):
	movl	-10(%rdi), %eax
	movl	-10(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(6bytes):
	movl	-6(%rdi), %eax
	movl	-6(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(2bytes):
	movzwl	-2(%rdi), %eax
	movzwl	-2(%rsi), %ecx
	cmpb	%cl, %al
	jne	L(set)
	cmp	%ecx, %eax
	jne	L(set)
	xor	%eax, %eax
	ret

	.p2align 4
L(47bytes):
	movl	-47(%rdi), %eax
	movl	-47(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(43bytes):
	movl	-43(%rdi), %eax
	movl	-43(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(39bytes):
	movl	-39(%rdi), %eax
	movl	-39(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(35bytes):
	movl	-35(%rdi), %eax
	movl	-35(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(31bytes):
	movl	-31(%rdi), %eax
	movl	-31(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(27bytes):
	movl	-27(%rdi), %eax
	movl	-27(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(23bytes):
	movl	-23(%rdi), %eax
	movl	-23(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(19bytes):
	movl	-19(%rdi), %eax
	movl	-19(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(15bytes):
	movl	-15(%rdi), %eax
	movl	-15(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(11bytes):
	movl	-11(%rdi), %eax
	movl	-11(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(7bytes):
	movl	-7(%rdi), %eax
	movl	-7(%rsi), %ecx
	cmp	%ecx, %eax
	jne	L(find_diff)
L(3bytes):
	movzwl	-3(%rdi), %eax
	movzwl	-3(%rsi), %ecx
	cmpb	%cl, %al
	jne	L(set)
	cmp	%ecx, %eax
	jne	L(set)
	movzbl	-1(%rdi), %eax
	cmpb	-1(%rsi), %al
	jne	L(set)
	xor	%eax, %eax
	ret

	.p2align 4
L(find_diff):
	cmpb	%cl, %al
	jne	L(set)
	cmpw	%cx, %ax
	jne	L(set)
	shr	$16, %eax
	shr	$16, %ecx
	cmpb	%cl, %al
	jne	L(set)

/* We get there only if we already know there is a
difference.  */

	cmp	%ecx, %eax
L(set):
	sbb	%eax, %eax
	sbb	$-1, %eax
	ret
# else

/* for wmemcmp */
	.p2align 4
L(find_diff):
	mov	$1, %eax
	jg	L(find_diff_bigger)
	neg	%eax
	ret

	.p2align 4
L(find_diff_bigger):
	ret
# endif

	.p2align 4
L(equal):
	xor	%eax, %eax
	ret

END (MEMCMP)
#endif

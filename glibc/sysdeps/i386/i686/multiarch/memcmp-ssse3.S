/* memcmp with SSSE3, wmemcmp with SSSE3
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
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
#  define MEMCMP		__memcmp_ssse3
# endif

# define CFI_PUSH(REG)	\
	cfi_adjust_cfa_offset (4);	\
	cfi_rel_offset (REG, 0)

# define CFI_POP(REG)	\
	cfi_adjust_cfa_offset (-4);	\
	cfi_restore (REG)

# define PUSH(REG)	pushl REG; CFI_PUSH (REG)
# define POP(REG)	popl REG; CFI_POP (REG)

# define PARMS		4
# define BLK1		PARMS
# define BLK2		BLK1+4
# define LEN		BLK2+4
# define RETURN_END	POP (%edi); POP (%esi); POP (%ebx); ret
# define RETURN		RETURN_END; cfi_restore_state; cfi_remember_state

/* Warning!
           wmemcmp has to use SIGNED comparison for elements.
           memcmp has to use UNSIGNED comparison for elemnts.
*/

	atom_text_section
ENTRY (MEMCMP)
	movl	LEN(%esp), %ecx

# ifdef USE_AS_WMEMCMP
	shl	$2, %ecx
	test	%ecx, %ecx
	jz	L(zero)
# endif

	movl	BLK1(%esp), %eax
	cmp	$48, %ecx
	movl	BLK2(%esp), %edx
	jae	L(48bytesormore)

# ifndef USE_AS_WMEMCMP
	cmp	$1, %ecx
	jbe	L(less1bytes)
# endif

	PUSH	(%ebx)
	add	%ecx, %edx
	add	%ecx, %eax
	jmp	L(less48bytes)

	CFI_POP	(%ebx)

# ifndef USE_AS_WMEMCMP
	.p2align 4
L(less1bytes):
	jb	L(zero)
	movb	(%eax), %cl
	cmp	(%edx), %cl
	je	L(zero)
	mov	$1, %eax
	ja	L(1bytesend)
	neg	%eax
L(1bytesend):
	ret
# endif

	.p2align 4
L(zero):
	xor	%eax, %eax
	ret

	.p2align 4
L(48bytesormore):
	PUSH	(%ebx)
	PUSH	(%esi)
	PUSH	(%edi)
	cfi_remember_state
	movdqu	(%eax), %xmm3
	movdqu	(%edx), %xmm0
	movl	%eax, %edi
	movl	%edx, %esi
	pcmpeqb	%xmm0, %xmm3
	pmovmskb %xmm3, %edx
	lea	16(%edi), %edi

	sub	$0xffff, %edx
	lea	16(%esi), %esi
	jnz	L(less16bytes)
	mov	%edi, %edx
	and	$0xf, %edx
	xor	%edx, %edi
	sub	%edx, %esi
	add	%edx, %ecx
	mov	%esi, %edx
	and	$0xf, %edx
	jz	L(shr_0)
	xor	%edx, %esi

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
	cmp	$80, %ecx
	jae	L(shr_0_gobble)
	lea	-48(%ecx), %ecx
	xor	%eax, %eax
	movaps	(%esi), %xmm1
	pcmpeqb	(%edi), %xmm1
	movaps	16(%esi), %xmm2
	pcmpeqb	16(%edi), %xmm2
	pand	%xmm1, %xmm2
	pmovmskb %xmm2, %edx
	add	$32, %edi
	add	$32, %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_0_gobble):
	lea	-48(%ecx), %ecx
	movdqa	(%esi), %xmm0
	xor	%eax, %eax
	pcmpeqb	(%edi), %xmm0
	sub	$32, %ecx
	movdqa	16(%esi), %xmm2
	pcmpeqb	16(%edi), %xmm2
L(shr_0_gobble_loop):
	pand	%xmm0, %xmm2
	sub	$32, %ecx
	pmovmskb %xmm2, %edx
	movdqa	%xmm0, %xmm1
	movdqa	32(%esi), %xmm0
	movdqa	48(%esi), %xmm2
	sbb	$0xffff, %edx
	pcmpeqb	32(%edi), %xmm0
	pcmpeqb	48(%edi), %xmm2
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	jz	L(shr_0_gobble_loop)

	pand	%xmm0, %xmm2
	cmp	$0, %ecx
	jge	L(shr_0_gobble_loop_next)
	inc	%edx
	add	$32, %ecx
L(shr_0_gobble_loop_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm2, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

# ifndef USE_AS_WMEMCMP
	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_1):
	cmp	$80, %ecx
	lea	-48(%ecx), %ecx
	mov	%edx, %eax
	jae	L(shr_1_gobble)

	movdqa	16(%esi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$1,(%esi), %xmm1
	pcmpeqb	(%edi), %xmm1

	movdqa	32(%esi), %xmm3
	palignr	$1,%xmm2, %xmm3
	pcmpeqb	16(%edi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	1(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_1_gobble):
	sub	$32, %ecx
	movdqa	16(%esi), %xmm0
	palignr	$1,(%esi), %xmm0
	pcmpeqb	(%edi), %xmm0

	movdqa	32(%esi), %xmm3
	palignr	$1,16(%esi), %xmm3
	pcmpeqb	16(%edi), %xmm3

L(shr_1_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %ecx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%esi), %xmm3
	palignr	$1,48(%esi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%esi), %xmm0
	palignr	$1,32(%esi), %xmm0
	pcmpeqb	32(%edi), %xmm0
	lea	32(%esi), %esi
	pcmpeqb	48(%edi), %xmm3

	lea	32(%edi), %edi
	jz	L(shr_1_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %ecx
	jge	L(shr_1_gobble_next)
	inc	%edx
	add	$32, %ecx
L(shr_1_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	1(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)


	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_2):
	cmp	$80, %ecx
	lea	-48(%ecx), %ecx
	mov	%edx, %eax
	jae	L(shr_2_gobble)

	movdqa	16(%esi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$2,(%esi), %xmm1
	pcmpeqb	(%edi), %xmm1

	movdqa	32(%esi), %xmm3
	palignr	$2,%xmm2, %xmm3
	pcmpeqb	16(%edi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	2(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_2_gobble):
	sub	$32, %ecx
	movdqa	16(%esi), %xmm0
	palignr	$2,(%esi), %xmm0
	pcmpeqb	(%edi), %xmm0

	movdqa	32(%esi), %xmm3
	palignr	$2,16(%esi), %xmm3
	pcmpeqb	16(%edi), %xmm3

L(shr_2_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %ecx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%esi), %xmm3
	palignr	$2,48(%esi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%esi), %xmm0
	palignr	$2,32(%esi), %xmm0
	pcmpeqb	32(%edi), %xmm0
	lea	32(%esi), %esi
	pcmpeqb	48(%edi), %xmm3

	lea	32(%edi), %edi
	jz	L(shr_2_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %ecx
	jge	L(shr_2_gobble_next)
	inc	%edx
	add	$32, %ecx
L(shr_2_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	2(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_3):
	cmp	$80, %ecx
	lea	-48(%ecx), %ecx
	mov	%edx, %eax
	jae	L(shr_3_gobble)

	movdqa	16(%esi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$3,(%esi), %xmm1
	pcmpeqb	(%edi), %xmm1

	movdqa	32(%esi), %xmm3
	palignr	$3,%xmm2, %xmm3
	pcmpeqb	16(%edi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	3(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_3_gobble):
	sub	$32, %ecx
	movdqa	16(%esi), %xmm0
	palignr	$3,(%esi), %xmm0
	pcmpeqb	(%edi), %xmm0

	movdqa	32(%esi), %xmm3
	palignr	$3,16(%esi), %xmm3
	pcmpeqb	16(%edi), %xmm3

L(shr_3_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %ecx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%esi), %xmm3
	palignr	$3,48(%esi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%esi), %xmm0
	palignr	$3,32(%esi), %xmm0
	pcmpeqb	32(%edi), %xmm0
	lea	32(%esi), %esi
	pcmpeqb	48(%edi), %xmm3

	lea	32(%edi), %edi
	jz	L(shr_3_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %ecx
	jge	L(shr_3_gobble_next)
	inc	%edx
	add	$32, %ecx
L(shr_3_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	3(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)
# endif

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_4):
	cmp	$80, %ecx
	lea	-48(%ecx), %ecx
	mov	%edx, %eax
	jae	L(shr_4_gobble)

	movdqa	16(%esi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$4,(%esi), %xmm1
	pcmpeqb	(%edi), %xmm1

	movdqa	32(%esi), %xmm3
	palignr	$4,%xmm2, %xmm3
	pcmpeqb	16(%edi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	4(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_4_gobble):
	sub	$32, %ecx
	movdqa	16(%esi), %xmm0
	palignr	$4,(%esi), %xmm0
	pcmpeqb	(%edi), %xmm0

	movdqa	32(%esi), %xmm3
	palignr	$4,16(%esi), %xmm3
	pcmpeqb	16(%edi), %xmm3

L(shr_4_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %ecx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%esi), %xmm3
	palignr	$4,48(%esi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%esi), %xmm0
	palignr	$4,32(%esi), %xmm0
	pcmpeqb	32(%edi), %xmm0
	lea	32(%esi), %esi
	pcmpeqb	48(%edi), %xmm3

	lea	32(%edi), %edi
	jz	L(shr_4_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %ecx
	jge	L(shr_4_gobble_next)
	inc	%edx
	add	$32, %ecx
L(shr_4_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	4(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

# ifndef USE_AS_WMEMCMP
	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_5):
	cmp	$80, %ecx
	lea	-48(%ecx), %ecx
	mov	%edx, %eax
	jae	L(shr_5_gobble)

	movdqa	16(%esi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$5,(%esi), %xmm1
	pcmpeqb	(%edi), %xmm1

	movdqa	32(%esi), %xmm3
	palignr	$5,%xmm2, %xmm3
	pcmpeqb	16(%edi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	5(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_5_gobble):
	sub	$32, %ecx
	movdqa	16(%esi), %xmm0
	palignr	$5,(%esi), %xmm0
	pcmpeqb	(%edi), %xmm0

	movdqa	32(%esi), %xmm3
	palignr	$5,16(%esi), %xmm3
	pcmpeqb	16(%edi), %xmm3

L(shr_5_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %ecx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%esi), %xmm3
	palignr	$5,48(%esi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%esi), %xmm0
	palignr	$5,32(%esi), %xmm0
	pcmpeqb	32(%edi), %xmm0
	lea	32(%esi), %esi
	pcmpeqb	48(%edi), %xmm3

	lea	32(%edi), %edi
	jz	L(shr_5_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %ecx
	jge	L(shr_5_gobble_next)
	inc	%edx
	add	$32, %ecx
L(shr_5_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	5(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_6):
	cmp	$80, %ecx
	lea	-48(%ecx), %ecx
	mov	%edx, %eax
	jae	L(shr_6_gobble)

	movdqa	16(%esi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$6,(%esi), %xmm1
	pcmpeqb	(%edi), %xmm1

	movdqa	32(%esi), %xmm3
	palignr	$6,%xmm2, %xmm3
	pcmpeqb	16(%edi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	6(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_6_gobble):
	sub	$32, %ecx
	movdqa	16(%esi), %xmm0
	palignr	$6,(%esi), %xmm0
	pcmpeqb	(%edi), %xmm0

	movdqa	32(%esi), %xmm3
	palignr	$6,16(%esi), %xmm3
	pcmpeqb	16(%edi), %xmm3

L(shr_6_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %ecx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%esi), %xmm3
	palignr	$6,48(%esi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%esi), %xmm0
	palignr	$6,32(%esi), %xmm0
	pcmpeqb	32(%edi), %xmm0
	lea	32(%esi), %esi
	pcmpeqb	48(%edi), %xmm3

	lea	32(%edi), %edi
	jz	L(shr_6_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %ecx
	jge	L(shr_6_gobble_next)
	inc	%edx
	add	$32, %ecx
L(shr_6_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	6(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_7):
	cmp	$80, %ecx
	lea	-48(%ecx), %ecx
	mov	%edx, %eax
	jae	L(shr_7_gobble)

	movdqa	16(%esi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$7,(%esi), %xmm1
	pcmpeqb	(%edi), %xmm1

	movdqa	32(%esi), %xmm3
	palignr	$7,%xmm2, %xmm3
	pcmpeqb	16(%edi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	7(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_7_gobble):
	sub	$32, %ecx
	movdqa	16(%esi), %xmm0
	palignr	$7,(%esi), %xmm0
	pcmpeqb	(%edi), %xmm0

	movdqa	32(%esi), %xmm3
	palignr	$7,16(%esi), %xmm3
	pcmpeqb	16(%edi), %xmm3

L(shr_7_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %ecx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%esi), %xmm3
	palignr	$7,48(%esi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%esi), %xmm0
	palignr	$7,32(%esi), %xmm0
	pcmpeqb	32(%edi), %xmm0
	lea	32(%esi), %esi
	pcmpeqb	48(%edi), %xmm3

	lea	32(%edi), %edi
	jz	L(shr_7_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %ecx
	jge	L(shr_7_gobble_next)
	inc	%edx
	add	$32, %ecx
L(shr_7_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	7(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)
# endif

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_8):
	cmp	$80, %ecx
	lea	-48(%ecx), %ecx
	mov	%edx, %eax
	jae	L(shr_8_gobble)

	movdqa	16(%esi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$8,(%esi), %xmm1
	pcmpeqb	(%edi), %xmm1

	movdqa	32(%esi), %xmm3
	palignr	$8,%xmm2, %xmm3
	pcmpeqb	16(%edi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	8(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_8_gobble):
	sub	$32, %ecx
	movdqa	16(%esi), %xmm0
	palignr	$8,(%esi), %xmm0
	pcmpeqb	(%edi), %xmm0

	movdqa	32(%esi), %xmm3
	palignr	$8,16(%esi), %xmm3
	pcmpeqb	16(%edi), %xmm3

L(shr_8_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %ecx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%esi), %xmm3
	palignr	$8,48(%esi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%esi), %xmm0
	palignr	$8,32(%esi), %xmm0
	pcmpeqb	32(%edi), %xmm0
	lea	32(%esi), %esi
	pcmpeqb	48(%edi), %xmm3

	lea	32(%edi), %edi
	jz	L(shr_8_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %ecx
	jge	L(shr_8_gobble_next)
	inc	%edx
	add	$32, %ecx
L(shr_8_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	8(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

# ifndef USE_AS_WMEMCMP
	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_9):
	cmp	$80, %ecx
	lea	-48(%ecx), %ecx
	mov	%edx, %eax
	jae	L(shr_9_gobble)

	movdqa	16(%esi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$9,(%esi), %xmm1
	pcmpeqb	(%edi), %xmm1

	movdqa	32(%esi), %xmm3
	palignr	$9,%xmm2, %xmm3
	pcmpeqb	16(%edi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	9(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_9_gobble):
	sub	$32, %ecx
	movdqa	16(%esi), %xmm0
	palignr	$9,(%esi), %xmm0
	pcmpeqb	(%edi), %xmm0

	movdqa	32(%esi), %xmm3
	palignr	$9,16(%esi), %xmm3
	pcmpeqb	16(%edi), %xmm3

L(shr_9_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %ecx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%esi), %xmm3
	palignr	$9,48(%esi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%esi), %xmm0
	palignr	$9,32(%esi), %xmm0
	pcmpeqb	32(%edi), %xmm0
	lea	32(%esi), %esi
	pcmpeqb	48(%edi), %xmm3

	lea	32(%edi), %edi
	jz	L(shr_9_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %ecx
	jge	L(shr_9_gobble_next)
	inc	%edx
	add	$32, %ecx
L(shr_9_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	9(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_10):
	cmp	$80, %ecx
	lea	-48(%ecx), %ecx
	mov	%edx, %eax
	jae	L(shr_10_gobble)

	movdqa	16(%esi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$10, (%esi), %xmm1
	pcmpeqb	(%edi), %xmm1

	movdqa	32(%esi), %xmm3
	palignr	$10,%xmm2, %xmm3
	pcmpeqb	16(%edi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	10(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_10_gobble):
	sub	$32, %ecx
	movdqa	16(%esi), %xmm0
	palignr	$10, (%esi), %xmm0
	pcmpeqb	(%edi), %xmm0

	movdqa	32(%esi), %xmm3
	palignr	$10, 16(%esi), %xmm3
	pcmpeqb	16(%edi), %xmm3

L(shr_10_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %ecx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%esi), %xmm3
	palignr	$10,48(%esi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%esi), %xmm0
	palignr	$10,32(%esi), %xmm0
	pcmpeqb	32(%edi), %xmm0
	lea	32(%esi), %esi
	pcmpeqb	48(%edi), %xmm3

	lea	32(%edi), %edi
	jz	L(shr_10_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %ecx
	jge	L(shr_10_gobble_next)
	inc	%edx
	add	$32, %ecx
L(shr_10_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	10(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_11):
	cmp	$80, %ecx
	lea	-48(%ecx), %ecx
	mov	%edx, %eax
	jae	L(shr_11_gobble)

	movdqa	16(%esi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$11, (%esi), %xmm1
	pcmpeqb	(%edi), %xmm1

	movdqa	32(%esi), %xmm3
	palignr	$11, %xmm2, %xmm3
	pcmpeqb	16(%edi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	11(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_11_gobble):
	sub	$32, %ecx
	movdqa	16(%esi), %xmm0
	palignr	$11, (%esi), %xmm0
	pcmpeqb	(%edi), %xmm0

	movdqa	32(%esi), %xmm3
	palignr	$11, 16(%esi), %xmm3
	pcmpeqb	16(%edi), %xmm3

L(shr_11_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %ecx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%esi), %xmm3
	palignr	$11,48(%esi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%esi), %xmm0
	palignr	$11,32(%esi), %xmm0
	pcmpeqb	32(%edi), %xmm0
	lea	32(%esi), %esi
	pcmpeqb	48(%edi), %xmm3

	lea	32(%edi), %edi
	jz	L(shr_11_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %ecx
	jge	L(shr_11_gobble_next)
	inc	%edx
	add	$32, %ecx
L(shr_11_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	11(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)
# endif

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_12):
	cmp	$80, %ecx
	lea	-48(%ecx), %ecx
	mov	%edx, %eax
	jae	L(shr_12_gobble)

	movdqa	16(%esi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$12, (%esi), %xmm1
	pcmpeqb	(%edi), %xmm1

	movdqa	32(%esi), %xmm3
	palignr	$12, %xmm2, %xmm3
	pcmpeqb	16(%edi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	12(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_12_gobble):
	sub	$32, %ecx
	movdqa	16(%esi), %xmm0
	palignr	$12, (%esi), %xmm0
	pcmpeqb	(%edi), %xmm0

	movdqa	32(%esi), %xmm3
	palignr	$12, 16(%esi), %xmm3
	pcmpeqb	16(%edi), %xmm3

L(shr_12_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %ecx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%esi), %xmm3
	palignr	$12,48(%esi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%esi), %xmm0
	palignr	$12,32(%esi), %xmm0
	pcmpeqb	32(%edi), %xmm0
	lea	32(%esi), %esi
	pcmpeqb	48(%edi), %xmm3

	lea	32(%edi), %edi
	jz	L(shr_12_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %ecx
	jge	L(shr_12_gobble_next)
	inc	%edx
	add	$32, %ecx
L(shr_12_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	12(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

# ifndef USE_AS_WMEMCMP
	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_13):
	cmp	$80, %ecx
	lea	-48(%ecx), %ecx
	mov	%edx, %eax
	jae	L(shr_13_gobble)

	movdqa	16(%esi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$13, (%esi), %xmm1
	pcmpeqb	(%edi), %xmm1

	movdqa	32(%esi), %xmm3
	palignr	$13, %xmm2, %xmm3
	pcmpeqb	16(%edi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	13(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_13_gobble):
	sub	$32, %ecx
	movdqa	16(%esi), %xmm0
	palignr	$13, (%esi), %xmm0
	pcmpeqb	(%edi), %xmm0

	movdqa	32(%esi), %xmm3
	palignr	$13, 16(%esi), %xmm3
	pcmpeqb	16(%edi), %xmm3

L(shr_13_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %ecx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%esi), %xmm3
	palignr	$13,48(%esi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%esi), %xmm0
	palignr	$13,32(%esi), %xmm0
	pcmpeqb	32(%edi), %xmm0
	lea	32(%esi), %esi
	pcmpeqb	48(%edi), %xmm3

	lea	32(%edi), %edi
	jz	L(shr_13_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %ecx
	jge	L(shr_13_gobble_next)
	inc	%edx
	add	$32, %ecx
L(shr_13_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	13(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_14):
	cmp	$80, %ecx
	lea	-48(%ecx), %ecx
	mov	%edx, %eax
	jae	L(shr_14_gobble)

	movdqa	16(%esi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$14, (%esi), %xmm1
	pcmpeqb	(%edi), %xmm1

	movdqa	32(%esi), %xmm3
	palignr	$14, %xmm2, %xmm3
	pcmpeqb	16(%edi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	14(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_14_gobble):
	sub	$32, %ecx
	movdqa	16(%esi), %xmm0
	palignr	$14, (%esi), %xmm0
	pcmpeqb	(%edi), %xmm0

	movdqa	32(%esi), %xmm3
	palignr	$14, 16(%esi), %xmm3
	pcmpeqb	16(%edi), %xmm3

L(shr_14_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %ecx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%esi), %xmm3
	palignr	$14,48(%esi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%esi), %xmm0
	palignr	$14,32(%esi), %xmm0
	pcmpeqb	32(%edi), %xmm0
	lea	32(%esi), %esi
	pcmpeqb	48(%edi), %xmm3

	lea	32(%edi), %edi
	jz	L(shr_14_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %ecx
	jge	L(shr_14_gobble_next)
	inc	%edx
	add	$32, %ecx
L(shr_14_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	14(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_15):
	cmp	$80, %ecx
	lea	-48(%ecx), %ecx
	mov	%edx, %eax
	jae	L(shr_15_gobble)

	movdqa	16(%esi), %xmm1
	movdqa	%xmm1, %xmm2
	palignr	$15, (%esi), %xmm1
	pcmpeqb	(%edi), %xmm1

	movdqa	32(%esi), %xmm3
	palignr	$15, %xmm2, %xmm3
	pcmpeqb	16(%edi), %xmm3

	pand	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)
	lea	(%ecx, %edi,1), %eax
	lea	15(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(shr_15_gobble):
	sub	$32, %ecx
	movdqa	16(%esi), %xmm0
	palignr	$15, (%esi), %xmm0
	pcmpeqb	(%edi), %xmm0

	movdqa	32(%esi), %xmm3
	palignr	$15, 16(%esi), %xmm3
	pcmpeqb	16(%edi), %xmm3

L(shr_15_gobble_loop):
	pand	%xmm0, %xmm3
	sub	$32, %ecx
	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1

	movdqa	64(%esi), %xmm3
	palignr	$15,48(%esi), %xmm3
	sbb	$0xffff, %edx
	movdqa	48(%esi), %xmm0
	palignr	$15,32(%esi), %xmm0
	pcmpeqb	32(%edi), %xmm0
	lea	32(%esi), %esi
	pcmpeqb	48(%edi), %xmm3

	lea	32(%edi), %edi
	jz	L(shr_15_gobble_loop)
	pand	%xmm0, %xmm3

	cmp	$0, %ecx
	jge	L(shr_15_gobble_next)
	inc	%edx
	add	$32, %ecx
L(shr_15_gobble_next):
	test	%edx, %edx
	jnz	L(exit)

	pmovmskb %xmm3, %edx
	movdqa	%xmm0, %xmm1
	lea	32(%edi), %edi
	lea	32(%esi), %esi
	sub	$0xffff, %edx
	jnz	L(exit)

	lea	(%ecx, %edi,1), %eax
	lea	15(%ecx, %esi,1), %edx
	POP	(%edi)
	POP	(%esi)
	jmp	L(less48bytes)
# endif

	cfi_restore_state
	cfi_remember_state
	.p2align 4
L(exit):
	pmovmskb %xmm1, %ebx
	sub	$0xffff, %ebx
	jz	L(first16bytes)
	lea	-16(%esi), %esi
	lea	-16(%edi), %edi
	mov	%ebx, %edx

L(first16bytes):
	add	%eax, %esi
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
L(Byte23):
	movzbl	-9(%edi), %eax
	movzbl	-9(%esi), %edx
	sub	%edx, %eax
	RETURN

	.p2align 4
L(Byte16):
	movzbl	-16(%edi), %eax
	movzbl	-16(%esi), %edx
	sub	%edx, %eax
	RETURN

	.p2align 4
L(Byte17):
	movzbl	-15(%edi), %eax
	movzbl	-15(%esi), %edx
	sub	%edx, %eax
	RETURN

	.p2align 4
L(Byte18):
	movzbl	-14(%edi), %eax
	movzbl	-14(%esi), %edx
	sub	%edx, %eax
	RETURN

	.p2align 4
L(Byte19):
	movzbl	-13(%edi), %eax
	movzbl	-13(%esi), %edx
	sub	%edx, %eax
	RETURN

	.p2align 4
L(Byte20):
	movzbl	-12(%edi), %eax
	movzbl	-12(%esi), %edx
	sub	%edx, %eax
	RETURN

	.p2align 4
L(Byte21):
	movzbl	-11(%edi), %eax
	movzbl	-11(%esi), %edx
	sub	%edx, %eax
	RETURN

	.p2align 4
L(Byte22):
	movzbl	-10(%edi), %eax
	movzbl	-10(%esi), %edx
	sub	%edx, %eax
	RETURN

	.p2align 4
L(next_24_bytes):
	lea	8(%edi), %edi
	lea	8(%esi), %esi
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

	.p2align 4
L(Byte31):
	movzbl	-9(%edi), %eax
	movzbl	-9(%esi), %edx
	sub	%edx, %eax
	RETURN_END
# else

/* special for wmemcmp */
	xor	%eax, %eax
	test	%dl, %dl
	jz	L(next_two_double_words)
	and	$15, %dl
	jz	L(second_double_word)
	mov	-16(%edi), %eax
	cmp	-16(%esi), %eax
	jne	L(nequal)
	RETURN

	.p2align 4
L(second_double_word):
	mov	-12(%edi), %eax
	cmp	-12(%esi), %eax
	jne	L(nequal)
	RETURN

	.p2align 4
L(next_two_double_words):
	and	$15, %dh
	jz	L(fourth_double_word)
	mov	-8(%edi), %eax
	cmp	-8(%esi), %eax
	jne	L(nequal)
	RETURN

	.p2align 4
L(fourth_double_word):
	mov	-4(%edi), %eax
	cmp	-4(%esi), %eax
	jne	L(nequal)
	RETURN

	.p2align 4
L(nequal):
	mov	$1, %eax
	jg	L(nequal_bigger)
	neg	%eax
	RETURN

	.p2align 4
L(nequal_bigger):
	RETURN_END
# endif

	CFI_PUSH (%ebx)

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
L(less48bytes):
	cmp	$8, %ecx
	jae	L(more8bytes)
# ifndef USE_AS_WMEMCMP
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
	mov	-44(%eax), %ecx
	mov	-44(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(40bytes):
	mov	-40(%eax), %ecx
	mov	-40(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(36bytes):
	mov	-36(%eax), %ecx
	mov	-36(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(32bytes):
	mov	-32(%eax), %ecx
	mov	-32(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(28bytes):
	mov	-28(%eax), %ecx
	mov	-28(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(24bytes):
	mov	-24(%eax), %ecx
	mov	-24(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(20bytes):
	mov	-20(%eax), %ecx
	mov	-20(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(16bytes):
	mov	-16(%eax), %ecx
	mov	-16(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(12bytes):
	mov	-12(%eax), %ecx
	mov	-12(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(8bytes):
	mov	-8(%eax), %ecx
	mov	-8(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(4bytes):
	mov	-4(%eax), %ecx
	mov	-4(%edx), %ebx
	cmp	%ebx, %ecx
	mov	$0, %eax
	jne	L(find_diff)
	POP	(%ebx)
	ret
	CFI_PUSH (%ebx)
# else
	.p2align 4
L(44bytes):
	mov	-44(%eax), %ecx
	cmp	-44(%edx), %ecx
	jne	L(find_diff)
L(40bytes):
	mov	-40(%eax), %ecx
	cmp	-40(%edx), %ecx
	jne	L(find_diff)
L(36bytes):
	mov	-36(%eax), %ecx
	cmp	-36(%edx), %ecx
	jne	L(find_diff)
L(32bytes):
	mov	-32(%eax), %ecx
	cmp	-32(%edx), %ecx
	jne	L(find_diff)
L(28bytes):
	mov	-28(%eax), %ecx
	cmp	-28(%edx), %ecx
	jne	L(find_diff)
L(24bytes):
	mov	-24(%eax), %ecx
	cmp	-24(%edx), %ecx
	jne	L(find_diff)
L(20bytes):
	mov	-20(%eax), %ecx
	cmp	-20(%edx), %ecx
	jne	L(find_diff)
L(16bytes):
	mov	-16(%eax), %ecx
	cmp	-16(%edx), %ecx
	jne	L(find_diff)
L(12bytes):
	mov	-12(%eax), %ecx
	cmp	-12(%edx), %ecx
	jne	L(find_diff)
L(8bytes):
	mov	-8(%eax), %ecx
	cmp	-8(%edx), %ecx
	jne	L(find_diff)
L(4bytes):
	mov	-4(%eax), %ecx
	xor	%eax, %eax
	cmp	-4(%edx), %ecx
	jne	L(find_diff)
	POP	(%ebx)
	ret
	CFI_PUSH (%ebx)
# endif

# ifndef USE_AS_WMEMCMP

	.p2align 4
L(45bytes):
	mov	-45(%eax), %ecx
	mov	-45(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(41bytes):
	mov	-41(%eax), %ecx
	mov	-41(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(37bytes):
	mov	-37(%eax), %ecx
	mov	-37(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(33bytes):
	mov	-33(%eax), %ecx
	mov	-33(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(29bytes):
	mov	-29(%eax), %ecx
	mov	-29(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(25bytes):
	mov	-25(%eax), %ecx
	mov	-25(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(21bytes):
	mov	-21(%eax), %ecx
	mov	-21(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(17bytes):
	mov	-17(%eax), %ecx
	mov	-17(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(13bytes):
	mov	-13(%eax), %ecx
	mov	-13(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(9bytes):
	mov	-9(%eax), %ecx
	mov	-9(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(5bytes):
	mov	-5(%eax), %ecx
	mov	-5(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	movzbl	-1(%eax), %ecx
	cmp	-1(%edx), %cl
	mov	$0, %eax
	jne	L(end)
	POP	(%ebx)
	ret
	CFI_PUSH (%ebx)

	.p2align 4
L(46bytes):
	mov	-46(%eax), %ecx
	mov	-46(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(42bytes):
	mov	-42(%eax), %ecx
	mov	-42(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(38bytes):
	mov	-38(%eax), %ecx
	mov	-38(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(34bytes):
	mov	-34(%eax), %ecx
	mov	-34(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(30bytes):
	mov	-30(%eax), %ecx
	mov	-30(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(26bytes):
	mov	-26(%eax), %ecx
	mov	-26(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(22bytes):
	mov	-22(%eax), %ecx
	mov	-22(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(18bytes):
	mov	-18(%eax), %ecx
	mov	-18(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(14bytes):
	mov	-14(%eax), %ecx
	mov	-14(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(10bytes):
	mov	-10(%eax), %ecx
	mov	-10(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(6bytes):
	mov	-6(%eax), %ecx
	mov	-6(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(2bytes):
	movzwl	-2(%eax), %ecx
	movzwl	-2(%edx), %ebx
	cmp	%bl, %cl
	jne	L(end)
	cmp	%bh, %ch
	mov	$0, %eax
	jne	L(end)
	POP	(%ebx)
	ret
	CFI_PUSH (%ebx)

	.p2align 4
L(47bytes):
	movl	-47(%eax), %ecx
	movl	-47(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(43bytes):
	movl	-43(%eax), %ecx
	movl	-43(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(39bytes):
	movl	-39(%eax), %ecx
	movl	-39(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(35bytes):
	movl	-35(%eax), %ecx
	movl	-35(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(31bytes):
	movl	-31(%eax), %ecx
	movl	-31(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(27bytes):
	movl	-27(%eax), %ecx
	movl	-27(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(23bytes):
	movl	-23(%eax), %ecx
	movl	-23(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(19bytes):
	movl	-19(%eax), %ecx
	movl	-19(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(15bytes):
	movl	-15(%eax), %ecx
	movl	-15(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(11bytes):
	movl	-11(%eax), %ecx
	movl	-11(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(7bytes):
	movl	-7(%eax), %ecx
	movl	-7(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(3bytes):
	movzwl	-3(%eax), %ecx
	movzwl	-3(%edx), %ebx
	cmpb	%bl, %cl
	jne	L(end)
	cmp	%bx, %cx
	jne	L(end)
	movzbl	-1(%eax), %eax
	cmpb	-1(%edx), %al
	mov	$0, %eax
	jne	L(end)
	POP	(%ebx)
	ret
	CFI_PUSH (%ebx)

	.p2align 4
L(find_diff):
	cmpb	%bl, %cl
	jne	L(end)
	cmp	%bx, %cx
	jne	L(end)
	shr	$16,%ecx
	shr	$16,%ebx
	cmp	%bl, %cl
	jne	L(end)
	cmp	%bx, %cx

	.p2align 4
L(end):
	POP	(%ebx)
	mov	$1, %eax
	ja	L(bigger)
	neg	%eax
L(bigger):
	ret
# else

/* for wmemcmp */
	.p2align 4
L(find_diff):
	POP	(%ebx)
	mov	$1, %eax
	jg	L(find_diff_bigger)
	neg	%eax
	ret

	.p2align 4
L(find_diff_bigger):
	ret

# endif
END (MEMCMP)
#endif

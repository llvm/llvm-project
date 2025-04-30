/* strrchr with SSE2 with bsf and bsr
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

	.text
ENTRY (__strrchr_sse2_bsf)

	mov	STR1(%esp), %ecx
	movd	STR2(%esp), %xmm1

	PUSH	(%edi)
	pxor	%xmm2, %xmm2
	mov	%ecx, %edi
	punpcklbw %xmm1, %xmm1
	punpcklbw %xmm1, %xmm1
	/* ECX has OFFSET. */
	and	$63, %ecx
	cmp	$48, %ecx
	pshufd	$0, %xmm1, %xmm1
	ja	L(crosscashe)

/* unaligned string. */
	movdqu	(%edi), %xmm0
	pcmpeqb	%xmm0, %xmm2
	pcmpeqb	%xmm1, %xmm0
	/* Find where NULL is.  */
	pmovmskb %xmm2, %edx
	/* Check if there is a match.  */
	pmovmskb %xmm0, %eax

	test	%eax, %eax
	jnz	L(unaligned_match1)

	test	%edx, %edx
	jnz	L(return_null)

	and	$-16, %edi
	add	$16, %edi

	PUSH	(%esi)
	PUSH	(%ebx)

	xor	%ebx, %ebx
	jmp	L(loop)

	CFI_POP	(%esi)
	CFI_POP	(%ebx)

	.p2align 4
L(unaligned_return_value1):
	bsf	%edx, %ecx
	mov	$2, %edx
	shl	%cl, %edx
	sub	$1, %edx
	and	%edx, %eax
	jz	L(return_null)
	bsr	%eax, %eax
	add	%edi, %eax
	POP	(%edi)
	ret
	CFI_PUSH	(%edi)

	.p2align 4
L(unaligned_match1):
	test	%edx, %edx
	jnz	L(unaligned_return_value1)

	PUSH	(%esi)
	PUSH	(%ebx)

	mov	%eax, %ebx
	lea	16(%edi), %esi
	and	$-16, %edi
	add	$16, %edi
	jmp	L(loop)

	CFI_POP	(%esi)
	CFI_POP	(%ebx)

	.p2align 4
	L(crosscashe):
/* Hancle unaligned string.  */
	and	$15, %ecx
	and	$-16, %edi
	pxor	%xmm3, %xmm3
	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm0, %xmm3
	pcmpeqb	%xmm1, %xmm0
	/* Find where NULL is.  */
	pmovmskb %xmm3, %edx
	/* Check if there is a match.  */
	pmovmskb %xmm0, %eax
	/* Remove the leading bytes.  */
	shr	%cl, %edx
	shr	%cl, %eax

	test	%eax, %eax
	jnz	L(unaligned_match)

	test	%edx, %edx
	jnz	L(return_null)

	add	$16, %edi

	PUSH	(%esi)
	PUSH	(%ebx)

	xor	%ebx, %ebx
	jmp	L(loop)

	CFI_POP	(%esi)
	CFI_POP	(%ebx)

	.p2align 4
L(unaligned_return_value):
	add	%ecx, %edi
	bsf	%edx, %ecx
	mov	$2, %edx
	shl	%cl, %edx
	sub	$1, %edx
	and	%edx, %eax
	jz	L(return_null)
	bsr	%eax, %eax
	add	%edi, %eax
	POP	(%edi)
	ret
	CFI_PUSH	(%edi)

	.p2align 4
L(unaligned_match):
	test	%edx, %edx
	jnz	L(unaligned_return_value)

	PUSH	(%esi)
	PUSH	(%ebx)

	mov	%eax, %ebx
	add	$16, %edi
	lea	(%edi, %ecx), %esi

/* Loop start on aligned string.  */
	.p2align 4
L(loop):
	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm0, %xmm2
	add	$16, %edi
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm2, %ecx
	pmovmskb %xmm0, %eax
	or	%eax, %ecx
	jnz	L(matches)

	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm0, %xmm2
	add	$16, %edi
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm2, %ecx
	pmovmskb %xmm0, %eax
	or	%eax, %ecx
	jnz	L(matches)

	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm0, %xmm2
	add	$16, %edi
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm2, %ecx
	pmovmskb %xmm0, %eax
	or	%eax, %ecx
	jnz	L(matches)

	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm0, %xmm2
	add	$16, %edi
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm2, %ecx
	pmovmskb %xmm0, %eax
	or	%eax, %ecx
	jz	L(loop)

L(matches):
	test	%eax, %eax
	jnz	L(match)
L(return_value):
	test	%ebx, %ebx
	jz	L(return_null_1)
	bsr	%ebx, %eax
	add	%esi, %eax

	POP	(%ebx)
	POP	(%esi)

	sub	$16, %eax
	POP	(%edi)
	ret

	CFI_PUSH	(%edi)
	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(match):
	pmovmskb %xmm2, %ecx
	test	%ecx, %ecx
	jnz	L(return_value_1)
	mov	%eax, %ebx
	mov	%edi, %esi
	jmp	L(loop)

	.p2align 4
L(return_value_1):
	bsf	%ecx, %ecx
	mov	$2, %edx
	shl	%cl, %edx
	sub	$1, %edx
	and	%edx, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)

	bsr	%eax, %eax
	add	%edi, %eax
	sub	$16, %eax
	POP	(%edi)
	ret

	CFI_PUSH	(%edi)
/* Return NULL.  */
	.p2align 4
L(return_null):
	xor	%eax, %eax
	POP	(%edi)
	ret

	CFI_PUSH	(%edi)
	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)
/* Return NULL.  */
	.p2align 4
L(return_null_1):
	POP	(%ebx)
	POP	(%esi)
	POP	(%edi)
	xor	%eax, %eax
	ret

END (__strrchr_sse2_bsf)
#endif

/* strchr with SSE2 with bsf
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

# define PARMS  8
# define ENTRANCE PUSH(%edi)
# define RETURN  POP(%edi); ret; CFI_PUSH(%edi);

# define STR1  PARMS
# define STR2  STR1+4

	.text
ENTRY (__strchr_sse2_bsf)

	ENTRANCE
	mov	STR1(%esp), %ecx
	movd	STR2(%esp), %xmm1

	pxor	%xmm2, %xmm2
	mov	%ecx, %edi
	punpcklbw %xmm1, %xmm1
	punpcklbw %xmm1, %xmm1
	/* ECX has OFFSET. */
	and	$15, %ecx
	pshufd	$0, %xmm1, %xmm1
	je	L(loop)

/* Handle unaligned string.  */
	and	$-16, %edi
	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm0, %xmm2
	pcmpeqb	%xmm1, %xmm0
	/* Find where NULL is.  */
	pmovmskb %xmm2, %edx
	/* Check if there is a match.  */
	pmovmskb %xmm0, %eax
	/* Remove the leading bytes.  */
	sarl	%cl, %edx
	sarl	%cl, %eax
	test	%eax, %eax
	je	L(unaligned_no_match)
	/* Check which byte is a match.  */
	bsf	%eax, %eax
	/* Is there a NULL? */
	test	%edx, %edx
	je	L(unaligned_match)
	bsf	%edx, %edx
	cmpl	%edx, %eax
	/* Return NULL if NULL comes first.  */
	ja	L(return_null)
L(unaligned_match):
	add	%edi, %eax
	add	%ecx, %eax
	RETURN

	.p2align 4
L(unaligned_no_match):
	test	%edx, %edx
	jne	L(return_null)
	pxor	%xmm2, %xmm2

	add	$16, %edi

	.p2align 4
/* Loop start on aligned string.  */
L(loop):
	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm0, %xmm2
	add	$16, %edi
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm2, %edx
	pmovmskb %xmm0, %eax
	or	%eax, %edx
	jnz	L(matches)

	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm0, %xmm2
	add	$16, %edi
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm2, %edx
	pmovmskb %xmm0, %eax
	or	%eax, %edx
	jnz	L(matches)

	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm0, %xmm2
	add	$16, %edi
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm2, %edx
	pmovmskb %xmm0, %eax
	or	%eax, %edx
	jnz	L(matches)

	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm0, %xmm2
	add	$16, %edi
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm2, %edx
	pmovmskb %xmm0, %eax
	or	%eax, %edx
	jnz	L(matches)
	jmp	L(loop)

L(matches):
	pmovmskb %xmm2, %edx
	test	%eax, %eax
	jz	L(return_null)
	bsf	%eax, %eax
	/* There is a match.  First find where NULL is.  */
	test	%edx, %edx
	je	L(match)
	bsf	%edx, %ecx
	/* Check if NULL comes first.  */
	cmpl	%ecx, %eax
	ja	L(return_null)
L(match):
	sub	$16, %edi
	add	%edi, %eax
	RETURN

/* Return NULL.  */
	.p2align 4
L(return_null):
	xor	%eax, %eax
	RETURN

END (__strchr_sse2_bsf)
#endif

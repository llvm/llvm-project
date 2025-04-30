/* strchr SSE2 without bsf
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

	atom_text_section
ENTRY (__strchr_sse2)

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
	jz	L(unaligned_no_match)
	/* Check which byte is a match.  */
	/* Is there a NULL? */
	add	%ecx, %edi
	test	%edx, %edx
	jz	L(match_case1)
	jmp	L(match_case2)

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
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm2, %edx
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches)
	test	%edx, %edx
	jnz	L(return_null)
	add	$16, %edi

	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm0, %xmm2
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm2, %edx
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches)
	test	%edx, %edx
	jnz	L(return_null)
	add	$16, %edi

	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm0, %xmm2
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm2, %edx
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches)
	test	%edx, %edx
	jnz	L(return_null)
	add	$16, %edi

	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm0, %xmm2
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm2, %edx
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches)
	test	%edx, %edx
	jnz	L(return_null)
	add	$16, %edi
	jmp	L(loop)

L(matches):
	/* There is a match.  First find where NULL is.  */
	test	%edx, %edx
	jz	L(match_case1)

	.p2align 4
L(match_case2):
	test	%al, %al
	jz	L(match_higth_case2)

	mov	%al, %cl
	and	$15, %cl
	jnz	L(match_case2_4)

	mov	%dl, %ch
	and	$15, %ch
	jnz	L(return_null)

	test	$0x10, %al
	jnz	L(Exit5)
	test	$0x10, %dl
	jnz	L(return_null)
	test	$0x20, %al
	jnz	L(Exit6)
	test	$0x20, %dl
	jnz	L(return_null)
	test	$0x40, %al
	jnz	L(Exit7)
	test	$0x40, %dl
	jnz	L(return_null)
	lea	7(%edi), %eax
	RETURN

	.p2align 4
L(match_case2_4):
	test	$0x01, %al
	jnz	L(Exit1)
	test	$0x01, %dl
	jnz	L(return_null)
	test	$0x02, %al
	jnz	L(Exit2)
	test	$0x02, %dl
	jnz	L(return_null)
	test	$0x04, %al
	jnz	L(Exit3)
	test	$0x04, %dl
	jnz	L(return_null)
	lea	3(%edi), %eax
	RETURN

	.p2align 4
L(match_higth_case2):
	test	%dl, %dl
	jnz	L(return_null)

	mov	%ah, %cl
	and	$15, %cl
	jnz	L(match_case2_12)

	mov	%dh, %ch
	and	$15, %ch
	jnz	L(return_null)

	test	$0x10, %ah
	jnz	L(Exit13)
	test	$0x10, %dh
	jnz	L(return_null)
	test	$0x20, %ah
	jnz	L(Exit14)
	test	$0x20, %dh
	jnz	L(return_null)
	test	$0x40, %ah
	jnz	L(Exit15)
	test	$0x40, %dh
	jnz	L(return_null)
	lea	15(%edi), %eax
	RETURN

	.p2align 4
L(match_case2_12):
	test	$0x01, %ah
	jnz	L(Exit9)
	test	$0x01, %dh
	jnz	L(return_null)
	test	$0x02, %ah
	jnz	L(Exit10)
	test	$0x02, %dh
	jnz	L(return_null)
	test	$0x04, %ah
	jnz	L(Exit11)
	test	$0x04, %dh
	jnz	L(return_null)
	lea	11(%edi), %eax
	RETURN

	.p2align 4
L(match_case1):
	test	%al, %al
	jz	L(match_higth_case1)

	test	$0x01, %al
	jnz	L(Exit1)
	test	$0x02, %al
	jnz	L(Exit2)
	test	$0x04, %al
	jnz	L(Exit3)
	test	$0x08, %al
	jnz	L(Exit4)
	test	$0x10, %al
	jnz	L(Exit5)
	test	$0x20, %al
	jnz	L(Exit6)
	test	$0x40, %al
	jnz	L(Exit7)
	lea	7(%edi), %eax
	RETURN

	.p2align 4
L(match_higth_case1):
	test	$0x01, %ah
	jnz	L(Exit9)
	test	$0x02, %ah
	jnz	L(Exit10)
	test	$0x04, %ah
	jnz	L(Exit11)
	test	$0x08, %ah
	jnz	L(Exit12)
	test	$0x10, %ah
	jnz	L(Exit13)
	test	$0x20, %ah
	jnz	L(Exit14)
	test	$0x40, %ah
	jnz	L(Exit15)
	lea	15(%edi), %eax
	RETURN

	.p2align 4
L(Exit1):
	lea	(%edi), %eax
	RETURN

	.p2align 4
L(Exit2):
	lea	1(%edi), %eax
	RETURN

	.p2align 4
L(Exit3):
	lea	2(%edi), %eax
	RETURN

	.p2align 4
L(Exit4):
	lea	3(%edi), %eax
	RETURN

	.p2align 4
L(Exit5):
	lea	4(%edi), %eax
	RETURN

	.p2align 4
L(Exit6):
	lea	5(%edi), %eax
	RETURN

	.p2align 4
L(Exit7):
	lea	6(%edi), %eax
	RETURN

	.p2align 4
L(Exit9):
	lea	8(%edi), %eax
	RETURN

	.p2align 4
L(Exit10):
	lea	9(%edi), %eax
	RETURN

	.p2align 4
L(Exit11):
	lea	10(%edi), %eax
	RETURN

	.p2align 4
L(Exit12):
	lea	11(%edi), %eax
	RETURN

	.p2align 4
L(Exit13):
	lea	12(%edi), %eax
	RETURN

	.p2align 4
L(Exit14):
	lea	13(%edi), %eax
	RETURN

	.p2align 4
L(Exit15):
	lea	14(%edi), %eax
	RETURN

/* Return NULL.  */
	.p2align 4
L(return_null):
	xor	%eax, %eax
	RETURN

END (__strchr_sse2)
#endif

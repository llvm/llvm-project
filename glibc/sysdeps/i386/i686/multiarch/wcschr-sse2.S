/* wcschr with SSE2, without using bsf instructions
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

# define PUSH(REG)	pushl REG; CFI_PUSH (REG)
# define POP(REG)	popl REG; CFI_POP (REG)

# define PARMS	4
# define STR1	PARMS
# define STR2	STR1+4

	atom_text_section
ENTRY (__wcschr_sse2)

	mov	STR1(%esp), %ecx
	movd	STR2(%esp), %xmm1

	mov	%ecx, %eax
	punpckldq %xmm1, %xmm1
	pxor	%xmm2, %xmm2
	punpckldq %xmm1, %xmm1

	and	$63, %eax
	cmp	$48, %eax
	ja	L(cross_cache)

	movdqu	(%ecx), %xmm0
	pcmpeqd	%xmm0, %xmm2
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm2, %edx
	pmovmskb %xmm0, %eax
	or	%eax, %edx
	jnz	L(matches)
	and	$-16, %ecx
	jmp	L(loop)

	.p2align 4
L(cross_cache):
	PUSH	(%edi)
	mov	%ecx, %edi
	mov	%eax, %ecx
	and	$-16, %edi
	and	$15, %ecx
	movdqa	(%edi), %xmm0
	pcmpeqd	%xmm0, %xmm2
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm2, %edx
	pmovmskb %xmm0, %eax

	sarl	%cl, %edx
	sarl	%cl, %eax
	test	%eax, %eax
	jz	L(unaligned_no_match)

	add	%edi, %ecx
	POP	(%edi)

	test	%edx, %edx
	jz	L(match_case1)
	test	%al, %al
	jz	L(match_higth_case2)
	test	$15, %al
	jnz	L(match_case2_4)
	test	$15, %dl
	jnz	L(return_null)
	lea	4(%ecx), %eax
	ret

	CFI_PUSH (%edi)

	.p2align 4
L(unaligned_no_match):
	mov	%edi, %ecx
	POP	(%edi)

	test	%edx, %edx
	jnz	L(return_null)

	pxor	%xmm2, %xmm2

/* Loop start on aligned string.  */
	.p2align 4
L(loop):
	add	$16, %ecx
	movdqa	(%ecx), %xmm0
	pcmpeqd	%xmm0, %xmm2
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm2, %edx
	pmovmskb %xmm0, %eax
	or	%eax, %edx
	jnz	L(matches)
	add	$16, %ecx

	movdqa	(%ecx), %xmm0
	pcmpeqd	%xmm0, %xmm2
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm2, %edx
	pmovmskb %xmm0, %eax
	or	%eax, %edx
	jnz	L(matches)
	add	$16, %ecx

	movdqa	(%ecx), %xmm0
	pcmpeqd	%xmm0, %xmm2
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm2, %edx
	pmovmskb %xmm0, %eax
	or	%eax, %edx
	jnz	L(matches)
	add	$16, %ecx

	movdqa	(%ecx), %xmm0
	pcmpeqd	%xmm0, %xmm2
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm2, %edx
	pmovmskb %xmm0, %eax
	or	%eax, %edx
	jz	L(loop)

	.p2align 4
L(matches):
	pmovmskb %xmm2, %edx
	test	%eax, %eax
	jz	L(return_null)
	test	%edx, %edx
	jz	L(match_case1)

	.p2align 4
L(match_case2):
	test	%al, %al
	jz	L(match_higth_case2)
	test	$15, %al
	jnz	L(match_case2_4)
	test	$15, %dl
	jnz	L(return_null)
	lea	4(%ecx), %eax
	ret

	.p2align 4
L(match_case2_4):
	mov	%ecx, %eax
	ret

	.p2align 4
L(match_higth_case2):
	test	%dl, %dl
	jnz	L(return_null)
	test	$15, %ah
	jnz	L(match_case2_12)
	test	$15, %dh
	jnz	L(return_null)
	lea	12(%ecx), %eax
	ret

	.p2align 4
L(match_case2_12):
	lea	8(%ecx), %eax
	ret

	.p2align 4
L(match_case1):
	test	%al, %al
	jz	L(match_higth_case1)

	test	$0x01, %al
	jnz	L(exit0)
	lea	4(%ecx), %eax
	ret

	.p2align 4
L(match_higth_case1):
	test	$0x01, %ah
	jnz	L(exit3)
	lea	12(%ecx), %eax
	ret

	.p2align 4
L(exit0):
	mov	%ecx, %eax
	ret

	.p2align 4
L(exit3):
	lea	8(%ecx), %eax
	ret

	.p2align 4
L(return_null):
	xor	%eax, %eax
	ret

END (__wcschr_sse2)
#endif

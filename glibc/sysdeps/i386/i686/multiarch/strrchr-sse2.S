/* strrchr SSE2 without bsf and bsr
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
# define ENTRANCE PUSH(%edi);
# define RETURN  POP(%edi); ret; CFI_PUSH(%edi);

# define STR1  PARMS
# define STR2  STR1+4

	atom_text_section
ENTRY (__strrchr_sse2)

	ENTRANCE
	mov	STR1(%esp), %ecx
	movd	STR2(%esp), %xmm1

	pxor	%xmm2, %xmm2
	mov	%ecx, %edi
	punpcklbw %xmm1, %xmm1
	punpcklbw %xmm1, %xmm1
	/* ECX has OFFSET. */
	and	$63, %ecx
	cmp	$48, %ecx
	pshufd	$0, %xmm1, %xmm1
	ja	L(crosscache)

/* unaligned string. */
	movdqu	(%edi), %xmm0
	pcmpeqb	%xmm0, %xmm2
	pcmpeqb	%xmm1, %xmm0
	/* Find where NULL is.  */
	pmovmskb %xmm2, %ecx
	/* Check if there is a match.  */
	pmovmskb %xmm0, %eax
	add	$16, %edi

	test	%eax, %eax
	jnz	L(unaligned_match1)

	test	%ecx, %ecx
	jnz	L(return_null)

	and	$-16, %edi

	PUSH	(%esi)
	PUSH	(%ebx)

	xor	%ebx, %ebx
	jmp	L(loop)

	CFI_POP	(%esi)
	CFI_POP	(%ebx)

	.p2align 4
L(unaligned_match1):
	test	%ecx, %ecx
	jnz	L(prolog_find_zero_1)

	PUSH	(%esi)
	PUSH	(%ebx)

	mov	%eax, %ebx
	mov	%edi, %esi
	and	$-16, %edi
	jmp	L(loop)

	CFI_POP	(%esi)
	CFI_POP	(%ebx)

	.p2align 4
L(crosscache):
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
	add	$16, %edi

	test	%eax, %eax
	jnz	L(unaligned_match)

	test	%edx, %edx
	jnz	L(return_null)

	PUSH	(%esi)
	PUSH	(%ebx)

	xor	%ebx, %ebx
	jmp	L(loop)

	CFI_POP	(%esi)
	CFI_POP	(%ebx)

	.p2align 4
L(unaligned_match):
	test	%edx, %edx
	jnz	L(prolog_find_zero)

	PUSH	(%esi)
	PUSH	(%ebx)

	mov	%eax, %ebx
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
	mov	%ebx, %eax
	mov	%esi, %edi

	POP	(%ebx)
	POP	(%esi)

	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(return_null_1):
	POP	(%ebx)
	POP	(%esi)

	xor	%eax, %eax
	RETURN

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(match):
	pmovmskb %xmm2, %ecx
	test	%ecx, %ecx
	jnz	L(find_zero)
	mov	%eax, %ebx
	mov	%edi, %esi
	jmp	L(loop)

	.p2align 4
L(find_zero):
	test	%cl, %cl
	jz	L(find_zero_high)
	mov	%cl, %dl
	and	$15, %dl
	jz	L(find_zero_8)
	test	$0x01, %cl
	jnz	L(FindZeroExit1)
	test	$0x02, %cl
	jnz	L(FindZeroExit2)
	test	$0x04, %cl
	jnz	L(FindZeroExit3)
	and	$1 << 4 - 1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)
	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(find_zero_8):
	test	$0x10, %cl
	jnz	L(FindZeroExit5)
	test	$0x20, %cl
	jnz	L(FindZeroExit6)
	test	$0x40, %cl
	jnz	L(FindZeroExit7)
	and	$1 << 8 - 1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)
	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(find_zero_high):
	mov	%ch, %dh
	and	$15, %dh
	jz	L(find_zero_high_8)
	test	$0x01, %ch
	jnz	L(FindZeroExit9)
	test	$0x02, %ch
	jnz	L(FindZeroExit10)
	test	$0x04, %ch
	jnz	L(FindZeroExit11)
	and	$1 << 12 - 1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)
	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(find_zero_high_8):
	test	$0x10, %ch
	jnz	L(FindZeroExit13)
	test	$0x20, %ch
	jnz	L(FindZeroExit14)
	test	$0x40, %ch
	jnz	L(FindZeroExit15)
	and	$1 << 16 - 1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)
	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(FindZeroExit1):
	and	$1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)
	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(FindZeroExit2):
	and	$1 << 2 - 1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)
	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(FindZeroExit3):
	and	$1 << 3 - 1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)
	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(FindZeroExit5):
	and	$1 << 5 - 1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)
	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(FindZeroExit6):
	and	$1 << 6 - 1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)
	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(FindZeroExit7):
	and	$1 << 7 - 1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)
	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(FindZeroExit9):
	and	$1 << 9 - 1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)
	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(FindZeroExit10):
	and	$1 << 10 - 1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)
	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(FindZeroExit11):
	and	$1 << 11 - 1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)
	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(FindZeroExit13):
	and	$1 << 13 - 1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)
	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(FindZeroExit14):
	and	$1 << 14 - 1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)
	jmp	L(match_exit)

	CFI_PUSH	(%ebx)
	CFI_PUSH	(%esi)

	.p2align 4
L(FindZeroExit15):
	and	$1 << 15 - 1, %eax
	jz	L(return_value)

	POP	(%ebx)
	POP	(%esi)

	.p2align 4
L(match_exit):
	test	%ah, %ah
	jnz	L(match_exit_high)
	mov	%al, %dl
	and	$15 << 4, %dl
	jnz	L(match_exit_8)
	test	$0x08, %al
	jnz	L(Exit4)
	test	$0x04, %al
	jnz	L(Exit3)
	test	$0x02, %al
	jnz	L(Exit2)
	lea	-16(%edi), %eax
	RETURN

	.p2align 4
L(match_exit_8):
	test	$0x80, %al
	jnz	L(Exit8)
	test	$0x40, %al
	jnz	L(Exit7)
	test	$0x20, %al
	jnz	L(Exit6)
	lea	-12(%edi), %eax
	RETURN

	.p2align 4
L(match_exit_high):
	mov	%ah, %dh
	and	$15 << 4, %dh
	jnz	L(match_exit_high_8)
	test	$0x08, %ah
	jnz	L(Exit12)
	test	$0x04, %ah
	jnz	L(Exit11)
	test	$0x02, %ah
	jnz	L(Exit10)
	lea	-8(%edi), %eax
	RETURN

	.p2align 4
L(match_exit_high_8):
	test	$0x80, %ah
	jnz	L(Exit16)
	test	$0x40, %ah
	jnz	L(Exit15)
	test	$0x20, %ah
	jnz	L(Exit14)
	lea	-4(%edi), %eax
	RETURN

	.p2align 4
L(Exit2):
	lea	-15(%edi), %eax
	RETURN

	.p2align 4
L(Exit3):
	lea	-14(%edi), %eax
	RETURN

	.p2align 4
L(Exit4):
	lea	-13(%edi), %eax
	RETURN

	.p2align 4
L(Exit6):
	lea	-11(%edi), %eax
	RETURN

	.p2align 4
L(Exit7):
	lea	-10(%edi), %eax
	RETURN

	.p2align 4
L(Exit8):
	lea	-9(%edi), %eax
	RETURN

	.p2align 4
L(Exit10):
	lea	-7(%edi), %eax
	RETURN

	.p2align 4
L(Exit11):
	lea	-6(%edi), %eax
	RETURN

	.p2align 4
L(Exit12):
	lea	-5(%edi), %eax
	RETURN

	.p2align 4
L(Exit14):
	lea	-3(%edi), %eax
	RETURN

	.p2align 4
L(Exit15):
	lea	-2(%edi), %eax
	RETURN

	.p2align 4
L(Exit16):
	lea	-1(%edi), %eax
	RETURN

/* Return NULL.  */
	.p2align 4
L(return_null):
	xor	%eax, %eax
	RETURN

	.p2align 4
L(prolog_find_zero):
	add	%ecx, %edi
	mov     %edx, %ecx
L(prolog_find_zero_1):
	test	%cl, %cl
	jz	L(prolog_find_zero_high)
	mov	%cl, %dl
	and	$15, %dl
	jz	L(prolog_find_zero_8)
	test	$0x01, %cl
	jnz	L(PrologFindZeroExit1)
	test	$0x02, %cl
	jnz	L(PrologFindZeroExit2)
	test	$0x04, %cl
	jnz	L(PrologFindZeroExit3)
	and	$1 << 4 - 1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(prolog_find_zero_8):
	test	$0x10, %cl
	jnz	L(PrologFindZeroExit5)
	test	$0x20, %cl
	jnz	L(PrologFindZeroExit6)
	test	$0x40, %cl
	jnz	L(PrologFindZeroExit7)
	and	$1 << 8 - 1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(prolog_find_zero_high):
	mov	%ch, %dh
	and	$15, %dh
	jz	L(prolog_find_zero_high_8)
	test	$0x01, %ch
	jnz	L(PrologFindZeroExit9)
	test	$0x02, %ch
	jnz	L(PrologFindZeroExit10)
	test	$0x04, %ch
	jnz	L(PrologFindZeroExit11)
	and	$1 << 12 - 1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(prolog_find_zero_high_8):
	test	$0x10, %ch
	jnz	L(PrologFindZeroExit13)
	test	$0x20, %ch
	jnz	L(PrologFindZeroExit14)
	test	$0x40, %ch
	jnz	L(PrologFindZeroExit15)
	and	$1 << 16 - 1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(PrologFindZeroExit1):
	and	$1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(PrologFindZeroExit2):
	and	$1 << 2 - 1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(PrologFindZeroExit3):
	and	$1 << 3 - 1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(PrologFindZeroExit5):
	and	$1 << 5 - 1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(PrologFindZeroExit6):
	and	$1 << 6 - 1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(PrologFindZeroExit7):
	and	$1 << 7 - 1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(PrologFindZeroExit9):
	and	$1 << 9 - 1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(PrologFindZeroExit10):
	and	$1 << 10 - 1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(PrologFindZeroExit11):
	and	$1 << 11 - 1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(PrologFindZeroExit13):
	and	$1 << 13 - 1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(PrologFindZeroExit14):
	and	$1 << 14 - 1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(PrologFindZeroExit15):
	and	$1 << 15 - 1, %eax
	jnz	L(match_exit)
	xor	%eax, %eax
	RETURN

END (__strrchr_sse2)
#endif

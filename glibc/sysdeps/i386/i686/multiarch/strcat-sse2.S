/* strcat with SSE2
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

# ifdef PIC
#  define JMPTBL(I, B) I - B

/* Load an entry in a jump table into ECX and branch to it.  TABLE is a
	jump table with relative offsets.  INDEX is a register contains the
	index into the jump table.   SCALE is the scale of INDEX. */

#  define BRANCH_TO_JMPTBL_ENTRY(TABLE, INDEX, SCALE)	\
	/* We first load PC into ECX.  */	\
	SETUP_PIC_REG(cx);	\
	/* Get the address of the jump table.  */	\
	addl	$(TABLE - .), %ecx;	\
	/* Get the entry and convert the relative offset to the	\
	absolute address.  */	\
	addl	(%ecx,INDEX,SCALE), %ecx;	\
	/* We loaded the jump table and adjusted ECX. Go.  */	\
	_CET_NOTRACK jmp *%ecx
# else
#  define JMPTBL(I, B) I

/* Branch to an entry in a jump table.  TABLE is a jump table with
	absolute offsets.  INDEX is a register contains the index into the
	jump table.  SCALE is the scale of INDEX. */

#  define BRANCH_TO_JMPTBL_ENTRY(TABLE, INDEX, SCALE)	\
	_CET_NOTRACK jmp *TABLE(,INDEX,SCALE)
# endif

# ifndef STRCAT
#  define STRCAT  __strcat_sse2
# endif

# define PARMS  4
# define STR1  PARMS+4
# define STR2  STR1+4

# ifdef USE_AS_STRNCAT
#  define LEN    STR2+8
#  define STR3   STR1+4
# else
#  define STR3   STR1
# endif

# define USE_AS_STRCAT
# ifdef USE_AS_STRNCAT
#  define RETURN  POP(%ebx); POP(%esi); ret; CFI_PUSH(%ebx); CFI_PUSH(%esi);
# else
#  define RETURN  POP(%esi); ret; CFI_PUSH(%esi);
# endif

.text
ENTRY (STRCAT)
	PUSH	(%esi)
	mov	STR1(%esp), %eax
	mov	STR2(%esp), %esi
# ifdef USE_AS_STRNCAT
	PUSH	(%ebx)
	movl	LEN(%esp), %ebx
	test	%ebx, %ebx
	jz	L(ExitZero)
# endif
	cmpb	$0, (%esi)
	mov	%esi, %ecx
	mov	%eax, %edx
	jz	L(ExitZero)

	and	$63, %ecx
	and	$63, %edx
	cmp	$32, %ecx
	ja	L(StrlenCore7_1)
	cmp	$48, %edx
	ja	L(alignment_prolog)

	pxor	%xmm0, %xmm0
	pxor	%xmm4, %xmm4
	pxor	%xmm7, %xmm7
	movdqu	(%eax), %xmm1
	movdqu	(%esi), %xmm5
	pcmpeqb	%xmm1, %xmm0
	movdqu	16(%esi), %xmm6
	pmovmskb %xmm0, %ecx
	pcmpeqb	%xmm5, %xmm4
	pcmpeqb	%xmm6, %xmm7
	test	%ecx, %ecx
	jnz	L(exit_less16_)
	mov	%eax, %ecx
	and	$-16, %eax
	jmp	L(loop_prolog)

L(alignment_prolog):
	pxor	%xmm0, %xmm0
	pxor	%xmm4, %xmm4
	mov	%edx, %ecx
	pxor	%xmm7, %xmm7
	and	$15, %ecx
	and	$-16, %eax
	pcmpeqb	(%eax), %xmm0
	movdqu	(%esi), %xmm5
	movdqu	16(%esi), %xmm6
	pmovmskb %xmm0, %edx
	pcmpeqb	%xmm5, %xmm4
	shr	%cl, %edx
	pcmpeqb	%xmm6, %xmm7
	test	%edx, %edx
	jnz	L(exit_less16)
	add	%eax, %ecx

	pxor	%xmm0, %xmm0
L(loop_prolog):
	pxor	%xmm1, %xmm1
	pxor	%xmm2, %xmm2
	pxor	%xmm3, %xmm3
	.p2align 4
L(align16_loop):
	pcmpeqb	16(%eax), %xmm0
	pmovmskb %xmm0, %edx
	test	%edx, %edx
	jnz	L(exit16)

	pcmpeqb	32(%eax), %xmm1
	pmovmskb %xmm1, %edx
	test	%edx, %edx
	jnz	L(exit32)

	pcmpeqb	48(%eax), %xmm2
	pmovmskb %xmm2, %edx
	test	%edx, %edx
	jnz	L(exit48)

	pcmpeqb	64(%eax), %xmm3
	pmovmskb %xmm3, %edx
	lea	64(%eax), %eax
	test	%edx, %edx
	jz	L(align16_loop)
	bsf	%edx, %edx
	add	%edx, %eax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit16):
	bsf	%edx, %edx
	lea	16(%eax, %edx), %eax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit32):
	bsf	%edx, %edx
	lea	32(%eax, %edx), %eax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit48):
	bsf	%edx, %edx
	lea	48(%eax, %edx), %eax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit_less16):
	bsf	%edx, %edx
	add	%ecx, %eax
	add	%edx, %eax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit_less16_):
	bsf	%ecx, %ecx
	add	%ecx, %eax

	.p2align 4
L(StartStrcpyPart):
	pmovmskb %xmm4, %edx
# ifdef USE_AS_STRNCAT
	cmp	$16, %ebx
	jbe	L(CopyFrom1To16BytesTail1Case2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesTail1)

	movdqu	%xmm5, (%eax)
	pmovmskb %xmm7, %edx
# ifdef USE_AS_STRNCAT
	cmp	$32, %ebx
	jbe	L(CopyFrom1To32Bytes1Case2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyFrom1To32Bytes1)

	mov	%esi, %ecx
	and	$-16, %esi
	and	$15, %ecx
	pxor	%xmm0, %xmm0
# ifdef USE_AS_STRNCAT
	add	%ecx, %ebx
	sbb	%edx, %edx
	or	%edx, %ebx
# endif
	sub	%ecx, %eax
	jmp	L(Unalign16Both)

L(StrlenCore7_1):
	mov	%eax, %ecx
	pxor	%xmm0, %xmm0
	and	$15, %ecx
	and	$-16, %eax
	pcmpeqb	(%eax), %xmm0
	pmovmskb %xmm0, %edx
	shr	%cl, %edx
	test	%edx, %edx
	jnz	L(exit_less16_1)
	add	%eax, %ecx

	pxor	%xmm0, %xmm0
	pxor	%xmm1, %xmm1
	pxor	%xmm2, %xmm2
	pxor	%xmm3, %xmm3

	.p2align 4
L(align16_loop_1):
	pcmpeqb	16(%eax), %xmm0
	pmovmskb %xmm0, %edx
	test	%edx, %edx
	jnz	L(exit16_1)

	pcmpeqb	32(%eax), %xmm1
	pmovmskb %xmm1, %edx
	test	%edx, %edx
	jnz	L(exit32_1)

	pcmpeqb	48(%eax), %xmm2
	pmovmskb %xmm2, %edx
	test	%edx, %edx
	jnz	L(exit48_1)

	pcmpeqb	64(%eax), %xmm3
	pmovmskb %xmm3, %edx
	lea	64(%eax), %eax
	test	%edx, %edx
	jz	L(align16_loop_1)
	bsf	%edx, %edx
	add	%edx, %eax
	jmp	L(StartStrcpyPart_1)

	.p2align 4
L(exit16_1):
	bsf	%edx, %edx
	lea	16(%eax, %edx), %eax
	jmp	L(StartStrcpyPart_1)

	.p2align 4
L(exit32_1):
	bsf	%edx, %edx
	lea	32(%eax, %edx), %eax
	jmp	L(StartStrcpyPart_1)

	.p2align 4
L(exit48_1):
	bsf	%edx, %edx
	lea	48(%eax, %edx), %eax
	jmp	L(StartStrcpyPart_1)

	.p2align 4
L(exit_less16_1):
	bsf	%edx, %edx
	add	%ecx, %eax
	add	%edx, %eax

	.p2align 4
L(StartStrcpyPart_1):
	mov	%esi, %ecx
	and	$15, %ecx
	and	$-16, %esi
	pxor	%xmm0, %xmm0
	pxor	%xmm1, %xmm1

# ifdef USE_AS_STRNCAT
	cmp	$48, %ebx
	ja      L(BigN)
# endif
	pcmpeqb	(%esi), %xmm1
# ifdef USE_AS_STRNCAT
	add	%ecx, %ebx
# endif
	pmovmskb %xmm1, %edx
	shr	%cl, %edx
# ifdef USE_AS_STRNCAT
	cmp	$16, %ebx
	jbe	L(CopyFrom1To16BytesTailCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesTail)

	pcmpeqb	16(%esi), %xmm0
	pmovmskb %xmm0, %edx
# ifdef USE_AS_STRNCAT
	cmp	$32, %ebx
	jbe	L(CopyFrom1To32BytesCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyFrom1To32Bytes)

	movdqu	(%esi, %ecx), %xmm1   /* copy 16 bytes */
	movdqu	%xmm1, (%eax)
	sub	%ecx, %eax

	.p2align 4
L(Unalign16Both):
	mov	$16, %ecx
	movdqa	(%esi, %ecx), %xmm1
	movaps	16(%esi, %ecx), %xmm2
	movdqu	%xmm1, (%eax, %ecx)
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %edx
	add	$16, %ecx
# ifdef USE_AS_STRNCAT
	sub	$48, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyFrom1To16Bytes)
L(Unalign16BothBigN):
	movaps	16(%esi, %ecx), %xmm3
	movdqu	%xmm2, (%eax, %ecx)
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %edx
	add	$16, %ecx
# ifdef USE_AS_STRNCAT
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%esi, %ecx), %xmm4
	movdqu	%xmm3, (%eax, %ecx)
	pcmpeqb	%xmm4, %xmm0
	pmovmskb %xmm0, %edx
	add	$16, %ecx
# ifdef USE_AS_STRNCAT
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%esi, %ecx), %xmm1
	movdqu	%xmm4, (%eax, %ecx)
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %edx
	add	$16, %ecx
# ifdef USE_AS_STRNCAT
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%esi, %ecx), %xmm2
	movdqu	%xmm1, (%eax, %ecx)
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %edx
	add	$16, %ecx
# ifdef USE_AS_STRNCAT
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%esi, %ecx), %xmm3
	movdqu	%xmm2, (%eax, %ecx)
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %edx
	add	$16, %ecx
# ifdef USE_AS_STRNCAT
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyFrom1To16Bytes)

	movdqu	%xmm3, (%eax, %ecx)
	mov	%esi, %edx
	lea	16(%esi, %ecx), %esi
	and	$-0x40, %esi
	sub	%esi, %edx
	sub	%edx, %eax
# ifdef USE_AS_STRNCAT
	lea	128(%ebx, %edx), %ebx
# endif
	movaps	(%esi), %xmm2
	movaps	%xmm2, %xmm4
	movaps	16(%esi), %xmm5
	movaps	32(%esi), %xmm3
	movaps	%xmm3, %xmm6
	movaps	48(%esi), %xmm7
	pminub	%xmm5, %xmm2
	pminub	%xmm7, %xmm3
	pminub	%xmm2, %xmm3
	pcmpeqb	%xmm0, %xmm3
	pmovmskb %xmm3, %edx
# ifdef USE_AS_STRNCAT
	sub	$64, %ebx
	jbe	L(UnalignedLeaveCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(Unaligned64Leave)

	.p2align 4
L(Unaligned64Loop_start):
	add	$64, %eax
	add	$64, %esi
	movdqu	%xmm4, -64(%eax)
	movaps	(%esi), %xmm2
	movdqa	%xmm2, %xmm4
	movdqu	%xmm5, -48(%eax)
	movaps	16(%esi), %xmm5
	pminub	%xmm5, %xmm2
	movaps	32(%esi), %xmm3
	movdqu	%xmm6, -32(%eax)
	movaps	%xmm3, %xmm6
	movdqu	%xmm7, -16(%eax)
	movaps	48(%esi), %xmm7
	pminub	%xmm7, %xmm3
	pminub	%xmm2, %xmm3
	pcmpeqb	%xmm0, %xmm3
	pmovmskb %xmm3, %edx
# ifdef USE_AS_STRNCAT
	sub	$64, %ebx
	jbe	L(UnalignedLeaveCase2OrCase3)
# endif
	test	%edx, %edx
	jz	L(Unaligned64Loop_start)

L(Unaligned64Leave):
	pxor	%xmm1, %xmm1

	pcmpeqb	%xmm4, %xmm0
	pcmpeqb	%xmm5, %xmm1
	pmovmskb %xmm0, %edx
	pmovmskb %xmm1, %ecx
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesUnaligned_0)
	test	%ecx, %ecx
	jnz	L(CopyFrom1To16BytesUnaligned_16)

	pcmpeqb	%xmm6, %xmm0
	pcmpeqb	%xmm7, %xmm1
	pmovmskb %xmm0, %edx
	pmovmskb %xmm1, %ecx
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesUnaligned_32)

	bsf	%ecx, %edx
	movdqu	%xmm4, (%eax)
	movdqu	%xmm5, 16(%eax)
	movdqu	%xmm6, 32(%eax)
	add	$48, %esi
	add	$48, %eax
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %edx, 4)

# ifdef USE_AS_STRNCAT
	.p2align 4
L(BigN):
	pcmpeqb	(%esi), %xmm1
	pmovmskb %xmm1, %edx
	shr	%cl, %edx
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesTail)

	pcmpeqb	16(%esi), %xmm0
	pmovmskb %xmm0, %edx
	test	%edx, %edx
	jnz	L(CopyFrom1To32Bytes)

	movdqu	(%esi, %ecx), %xmm1   /* copy 16 bytes */
	movdqu	%xmm1, (%eax)
	sub	%ecx, %eax
	sub     $48, %ebx
	add     %ecx, %ebx

	mov	$16, %ecx
	movdqa	(%esi, %ecx), %xmm1
	movaps	16(%esi, %ecx), %xmm2
	movdqu	%xmm1, (%eax, %ecx)
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %edx
	add	$16, %ecx
	test	%edx, %edx
	jnz	L(CopyFrom1To16Bytes)
	jmp	L(Unalign16BothBigN)
# endif

/*------------end of main part-------------------------------*/

/* Case1 */
	.p2align 4
L(CopyFrom1To16Bytes):
	add	%ecx, %eax
	add	%ecx, %esi
	bsf	%edx, %edx
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %edx, 4)

	.p2align 4
L(CopyFrom1To16BytesTail):
	add	%ecx, %esi
	bsf	%edx, %edx
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %edx, 4)

	.p2align 4
L(CopyFrom1To32Bytes1):
	add	$16, %esi
	add	$16, %eax
L(CopyFrom1To16BytesTail1):
	bsf	%edx, %edx
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %edx, 4)

	.p2align 4
L(CopyFrom1To32Bytes):
	bsf	%edx, %edx
	add	%ecx, %esi
	add	$16, %edx
	sub	%ecx, %edx
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %edx, 4)

	.p2align 4
L(CopyFrom1To16BytesUnaligned_0):
	bsf	%edx, %edx
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %edx, 4)

	.p2align 4
L(CopyFrom1To16BytesUnaligned_16):
	bsf	%ecx, %edx
	movdqu	%xmm4, (%eax)
	add	$16, %esi
	add	$16, %eax
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %edx, 4)

	.p2align 4
L(CopyFrom1To16BytesUnaligned_32):
	bsf	%edx, %edx
	movdqu	%xmm4, (%eax)
	movdqu	%xmm5, 16(%eax)
	add	$32, %esi
	add	$32, %eax
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %edx, 4)

# ifdef USE_AS_STRNCAT

	.p2align 4
L(CopyFrom1To16BytesExit):
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %edx, 4)

/* Case2 */

	.p2align 4
L(CopyFrom1To16BytesCase2):
	add	$16, %ebx
	add	%ecx, %eax
	add	%ecx, %esi
	bsf	%edx, %edx
	cmp	%ebx, %edx
	jb	L(CopyFrom1To16BytesExit)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncatTable), %ebx, 4)

	.p2align 4
L(CopyFrom1To32BytesCase2):
	sub	%ecx, %ebx
	add	%ecx, %esi
	bsf	%edx, %edx
	add	$16, %edx
	sub	%ecx, %edx
	cmp	%ebx, %edx
	jb	L(CopyFrom1To16BytesExit)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncatTable), %ebx, 4)

L(CopyFrom1To16BytesTailCase2):
	sub	%ecx, %ebx
	add	%ecx, %esi
	bsf	%edx, %edx
	cmp	%ebx, %edx
	jb	L(CopyFrom1To16BytesExit)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncatTable), %ebx, 4)

L(CopyFrom1To16BytesTail1Case2):
	bsf	%edx, %edx
	cmp	%ebx, %edx
	jb	L(CopyFrom1To16BytesExit)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncatTable), %ebx, 4)

/* Case2 or Case3,  Case3 */

	.p2align 4
L(CopyFrom1To16BytesCase2OrCase3):
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesCase2)
L(CopyFrom1To16BytesCase3):
	add	$16, %ebx
	add	%ecx, %eax
	add	%ecx, %esi
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncatTable), %ebx, 4)

	.p2align 4
L(CopyFrom1To32BytesCase2OrCase3):
	test	%edx, %edx
	jnz	L(CopyFrom1To32BytesCase2)
	sub	%ecx, %ebx
	add	%ecx, %esi
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncatTable), %ebx, 4)

	.p2align 4
L(CopyFrom1To16BytesTailCase2OrCase3):
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesTailCase2)
	sub	%ecx, %ebx
	add	%ecx, %esi
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncatTable), %ebx, 4)

	.p2align 4
L(CopyFrom1To32Bytes1Case2OrCase3):
	add	$16, %eax
	add	$16, %esi
	sub	$16, %ebx
L(CopyFrom1To16BytesTail1Case2OrCase3):
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesTail1Case2)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncatTable), %ebx, 4)

# endif

# ifdef USE_AS_STRNCAT
	.p2align 4
L(StrncatExit0):
	movb	%bh, (%eax)
	mov	STR3(%esp), %eax
	RETURN
# endif

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit1):
	movb	%bh, 1(%eax)
# endif
L(Exit1):
# ifdef USE_AS_STRNCAT
	movb	(%esi), %dh
# endif
	movb	%dh, (%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit2):
	movb	%bh, 2(%eax)
# endif
L(Exit2):
	movw	(%esi), %dx
	movw	%dx, (%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit3):
	movb	%bh, 3(%eax)
# endif
L(Exit3):
	movw	(%esi), %cx
	movw	%cx, (%eax)
# ifdef USE_AS_STRNCAT
	movb	2(%esi), %dh
# endif
	movb	%dh, 2(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit4):
	movb	%bh, 4(%eax)
# endif
L(Exit4):
	movl	(%esi), %edx
	movl	%edx, (%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit5):
	movb	%bh, 5(%eax)
# endif
L(Exit5):
	movl	(%esi), %ecx
# ifdef USE_AS_STRNCAT
	movb	4(%esi), %dh
# endif
	movb	%dh, 4(%eax)
	movl	%ecx, (%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit6):
	movb	%bh, 6(%eax)
# endif
L(Exit6):
	movl	(%esi), %ecx
	movw	4(%esi), %dx
	movl	%ecx, (%eax)
	movw	%dx, 4(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit7):
	movb	%bh, 7(%eax)
# endif
L(Exit7):
	movl	(%esi), %ecx
	movl	3(%esi), %edx
	movl	%ecx, (%eax)
	movl	%edx, 3(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit8):
	movb	%bh, 8(%eax)
# endif
L(Exit8):
	movlpd	(%esi), %xmm0
	movlpd	%xmm0, (%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit9):
	movb	%bh, 9(%eax)
# endif
L(Exit9):
	movlpd	(%esi), %xmm0
# ifdef USE_AS_STRNCAT
	movb	8(%esi), %dh
# endif
	movb	%dh, 8(%eax)
	movlpd	%xmm0, (%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit10):
	movb	%bh, 10(%eax)
# endif
L(Exit10):
	movlpd	(%esi), %xmm0
	movw	8(%esi), %dx
	movlpd	%xmm0, (%eax)
	movw	%dx, 8(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit11):
	movb	%bh, 11(%eax)
# endif
L(Exit11):
	movlpd	(%esi), %xmm0
	movl	7(%esi), %edx
	movlpd	%xmm0, (%eax)
	movl	%edx, 7(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit12):
	movb	%bh, 12(%eax)
# endif
L(Exit12):
	movlpd	(%esi), %xmm0
	movl	8(%esi), %edx
	movlpd	%xmm0, (%eax)
	movl	%edx, 8(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit13):
	movb	%bh, 13(%eax)
# endif
L(Exit13):
	movlpd	(%esi), %xmm0
	movlpd	5(%esi), %xmm1
	movlpd	%xmm0, (%eax)
	movlpd	%xmm1, 5(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit14):
	movb	%bh, 14(%eax)
# endif
L(Exit14):
	movlpd	(%esi), %xmm0
	movlpd	6(%esi), %xmm1
	movlpd	%xmm0, (%eax)
	movlpd	%xmm1, 6(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit15):
	movb	%bh, 15(%eax)
# endif
L(Exit15):
	movlpd	(%esi), %xmm0
	movlpd	7(%esi), %xmm1
	movlpd	%xmm0, (%eax)
	movlpd	%xmm1, 7(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit16):
	movb	%bh, 16(%eax)
# endif
L(Exit16):
	movdqu	(%esi), %xmm0
	movdqu	%xmm0, (%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit17):
	movb	%bh, 17(%eax)
# endif
L(Exit17):
	movdqu	(%esi), %xmm0
# ifdef USE_AS_STRNCAT
	movb	16(%esi), %dh
# endif
	movdqu	%xmm0, (%eax)
	movb	%dh, 16(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit18):
	movb	%bh, 18(%eax)
# endif
L(Exit18):
	movdqu	(%esi), %xmm0
	movw	16(%esi), %cx
	movdqu	%xmm0, (%eax)
	movw	%cx, 16(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit19):
	movb	%bh, 19(%eax)
# endif
L(Exit19):
	movdqu	(%esi), %xmm0
	movl	15(%esi), %ecx
	movdqu	%xmm0, (%eax)
	movl	%ecx, 15(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit20):
	movb	%bh, 20(%eax)
# endif
L(Exit20):
	movdqu	(%esi), %xmm0
	movl	16(%esi), %ecx
	movdqu	%xmm0, (%eax)
	movl	%ecx, 16(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit21):
	movb	%bh, 21(%eax)
# endif
L(Exit21):
	movdqu	(%esi), %xmm0
	movl	16(%esi), %ecx
# ifdef USE_AS_STRNCAT
	movb	20(%esi), %dh
# endif
	movdqu	%xmm0, (%eax)
	movl	%ecx, 16(%eax)
	movb	%dh, 20(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit22):
	movb	%bh, 22(%eax)
# endif
L(Exit22):
	movdqu	(%esi), %xmm0
	movlpd	14(%esi), %xmm3
	movdqu	%xmm0, (%eax)
	movlpd	%xmm3, 14(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit23):
	movb	%bh, 23(%eax)
# endif
L(Exit23):
	movdqu	(%esi), %xmm0
	movlpd	15(%esi), %xmm3
	movdqu	%xmm0, (%eax)
	movlpd	%xmm3, 15(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit24):
	movb	%bh, 24(%eax)
# endif
L(Exit24):
	movdqu	(%esi), %xmm0
	movlpd	16(%esi), %xmm2
	movdqu	%xmm0, (%eax)
	movlpd	%xmm2, 16(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit25):
	movb	%bh, 25(%eax)
# endif
L(Exit25):
	movdqu	(%esi), %xmm0
	movlpd	16(%esi), %xmm2
# ifdef USE_AS_STRNCAT
	movb	24(%esi), %dh
# endif
	movdqu	%xmm0, (%eax)
	movlpd	%xmm2, 16(%eax)
	movb	%dh, 24(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit26):
	movb	%bh, 26(%eax)
# endif
L(Exit26):
	movdqu	(%esi), %xmm0
	movlpd	16(%esi), %xmm2
	movw	24(%esi), %cx
	movdqu	%xmm0, (%eax)
	movlpd	%xmm2, 16(%eax)
	movw	%cx, 24(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit27):
	movb	%bh, 27(%eax)
# endif
L(Exit27):
	movdqu	(%esi), %xmm0
	movlpd	16(%esi), %xmm2
	movl	23(%esi), %ecx
	movdqu	%xmm0, (%eax)
	movlpd	%xmm2, 16(%eax)
	movl	%ecx, 23(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit28):
	movb	%bh, 28(%eax)
# endif
L(Exit28):
	movdqu	(%esi), %xmm0
	movlpd	16(%esi), %xmm2
	movl	24(%esi), %ecx
	movdqu	%xmm0, (%eax)
	movlpd	%xmm2, 16(%eax)
	movl	%ecx, 24(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit29):
	movb	%bh, 29(%eax)
# endif
L(Exit29):
	movdqu	(%esi), %xmm0
	movdqu	13(%esi), %xmm2
	movdqu	%xmm0, (%eax)
	movdqu	%xmm2, 13(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit30):
	movb	%bh, 30(%eax)
# endif
L(Exit30):
	movdqu	(%esi), %xmm0
	movdqu	14(%esi), %xmm2
	movdqu	%xmm0, (%eax)
	movdqu	%xmm2, 14(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit31):
	movb	%bh, 31(%eax)
# endif
L(Exit31):
	movdqu	(%esi), %xmm0
	movdqu	15(%esi), %xmm2
	movdqu	%xmm0, (%eax)
	movdqu	%xmm2, 15(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
# ifdef USE_AS_STRNCAT
L(StrncatExit32):
	movb	%bh, 32(%eax)
# endif
L(Exit32):
	movdqu	(%esi), %xmm0
	movdqu	16(%esi), %xmm2
	movdqu	%xmm0, (%eax)
	movdqu	%xmm2, 16(%eax)
	mov	STR3(%esp), %eax
	RETURN

# ifdef USE_AS_STRNCAT

	.p2align 4
L(UnalignedLeaveCase2OrCase3):
	test	%edx, %edx
	jnz	L(Unaligned64LeaveCase2)
L(Unaligned64LeaveCase3):
	lea	64(%ebx), %ecx
	and	$-16, %ecx
	add	$48, %ebx
	jl	L(CopyFrom1To16BytesCase3)
	movdqu	%xmm4, (%eax)
	sub	$16, %ebx
	jb	L(CopyFrom1To16BytesCase3)
	movdqu	%xmm5, 16(%eax)
	sub	$16, %ebx
	jb	L(CopyFrom1To16BytesCase3)
	movdqu	%xmm6, 32(%eax)
	sub	$16, %ebx
	jb	L(CopyFrom1To16BytesCase3)
	movdqu	%xmm7, 48(%eax)
	xor	%bh, %bh
	movb	%bh, 64(%eax)
	mov	STR3(%esp), %eax
	RETURN

	.p2align 4
L(Unaligned64LeaveCase2):
	xor	%ecx, %ecx
	pcmpeqb	%xmm4, %xmm0
	pmovmskb %xmm0, %edx
	add	$48, %ebx
	jle	L(CopyFrom1To16BytesCase2OrCase3)
	test	%edx, %edx
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqb	%xmm5, %xmm0
	pmovmskb %xmm0, %edx
	movdqu	%xmm4, (%eax)
	add	$16, %ecx
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%edx, %edx
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqb	%xmm6, %xmm0
	pmovmskb %xmm0, %edx
	movdqu	%xmm5, 16(%eax)
	add	$16, %ecx
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%edx, %edx
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqb	%xmm7, %xmm0
	pmovmskb %xmm0, %edx
	movdqu	%xmm6, 32(%eax)
	lea	16(%eax, %ecx), %eax
	lea	16(%esi, %ecx), %esi
	bsf	%edx, %edx
	cmp	%ebx, %edx
	jb	L(CopyFrom1To16BytesExit)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncatTable), %ebx, 4)
# endif
	.p2align 4
L(ExitZero):
	RETURN

END (STRCAT)

	.p2align 4
	.section .rodata
L(ExitTable):
	.int	JMPTBL(L(Exit1), L(ExitTable))
	.int	JMPTBL(L(Exit2), L(ExitTable))
	.int	JMPTBL(L(Exit3), L(ExitTable))
	.int	JMPTBL(L(Exit4), L(ExitTable))
	.int	JMPTBL(L(Exit5), L(ExitTable))
	.int	JMPTBL(L(Exit6), L(ExitTable))
	.int	JMPTBL(L(Exit7), L(ExitTable))
	.int	JMPTBL(L(Exit8), L(ExitTable))
	.int	JMPTBL(L(Exit9), L(ExitTable))
	.int	JMPTBL(L(Exit10), L(ExitTable))
	.int	JMPTBL(L(Exit11), L(ExitTable))
	.int	JMPTBL(L(Exit12), L(ExitTable))
	.int	JMPTBL(L(Exit13), L(ExitTable))
	.int	JMPTBL(L(Exit14), L(ExitTable))
	.int	JMPTBL(L(Exit15), L(ExitTable))
	.int	JMPTBL(L(Exit16), L(ExitTable))
	.int	JMPTBL(L(Exit17), L(ExitTable))
	.int	JMPTBL(L(Exit18), L(ExitTable))
	.int	JMPTBL(L(Exit19), L(ExitTable))
	.int	JMPTBL(L(Exit20), L(ExitTable))
	.int	JMPTBL(L(Exit21), L(ExitTable))
	.int	JMPTBL(L(Exit22), L(ExitTable))
	.int	JMPTBL(L(Exit23), L(ExitTable))
	.int	JMPTBL(L(Exit24), L(ExitTable))
	.int	JMPTBL(L(Exit25), L(ExitTable))
	.int	JMPTBL(L(Exit26), L(ExitTable))
	.int	JMPTBL(L(Exit27), L(ExitTable))
	.int	JMPTBL(L(Exit28), L(ExitTable))
	.int	JMPTBL(L(Exit29), L(ExitTable))
	.int	JMPTBL(L(Exit30), L(ExitTable))
	.int	JMPTBL(L(Exit31), L(ExitTable))
	.int	JMPTBL(L(Exit32), L(ExitTable))
# ifdef USE_AS_STRNCAT
L(ExitStrncatTable):
	.int	JMPTBL(L(StrncatExit0), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit1), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit2), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit3), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit4), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit5), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit6), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit7), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit8), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit9), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit10), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit11), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit12), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit13), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit14), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit15), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit16), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit17), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit18), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit19), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit20), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit21), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit22), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit23), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit24), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit25), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit26), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit27), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit28), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit29), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit30), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit31), L(ExitStrncatTable))
	.int	JMPTBL(L(StrncatExit32), L(ExitStrncatTable))
# endif
#endif

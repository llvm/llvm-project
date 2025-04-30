/* strcpy with SSE2 and unaligned load
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


# define CFI_PUSH(REG)                  \
	cfi_adjust_cfa_offset (4);     \
	cfi_rel_offset (REG, 0)

# define CFI_POP(REG)                   \
	cfi_adjust_cfa_offset (-4);    \
	cfi_restore (REG)

# define PUSH(REG) pushl REG; CFI_PUSH (REG)
# define POP(REG) popl REG; CFI_POP (REG)

# ifndef STRCPY
#  define STRCPY  __strcpy_sse2
# endif

# define STR1  PARMS
# define STR2  STR1+4
# define LEN  STR2+4

# ifdef USE_AS_STRNCPY
#  define PARMS  16
#  define ENTRANCE PUSH(%ebx); PUSH(%esi); PUSH(%edi)
#  define RETURN  POP(%edi); POP(%esi); POP(%ebx); ret;          \
	CFI_PUSH(%ebx); CFI_PUSH(%esi); CFI_PUSH(%edi);

# ifdef PIC
#  define JMPTBL(I, B)	I - B

/* Load an entry in a jump table into ECX and branch to it. TABLE is a
	jump table with relative offsets.
	INDEX is a register contains the index into the jump table.
	SCALE is the scale of INDEX. */

#  define BRANCH_TO_JMPTBL_ENTRY(TABLE, INDEX, SCALE)            \
	/* We first load PC into ECX.  */                       \
	SETUP_PIC_REG(cx);                                      \
	/* Get the address of the jump table.  */               \
	addl	$(TABLE - .), %ecx;                             \
	/* Get the entry and convert the relative offset to the \
	absolute	address.  */                            \
	addl	(%ecx,INDEX,SCALE), %ecx;                       \
	/* We loaded the jump table and adjusted ECX. Go.  */  \
	_CET_NOTRACK jmp *%ecx
# else
#  define JMPTBL(I, B)	I

/* Branch to an entry in a jump table.  TABLE is a jump table with
	absolute	offsets.  INDEX is a register contains the index into the
	jump	table.  SCALE is the scale of INDEX. */

#  define BRANCH_TO_JMPTBL_ENTRY(TABLE, INDEX, SCALE)		\
	_CET_NOTRACK jmp *TABLE(,INDEX,SCALE)
# endif

.text
ENTRY (STRCPY)
	ENTRANCE
	mov	STR1(%esp), %edi
	mov	STR2(%esp), %esi
	movl	LEN(%esp), %ebx
	test	%ebx, %ebx
	jz	L(ExitZero)

	mov	%esi, %ecx
# ifndef USE_AS_STPCPY
	mov	%edi, %eax      /* save result */
# endif
	and	$15, %ecx
	jz	L(SourceStringAlignmentZero)

	and	$-16, %esi
	pxor	%xmm0, %xmm0
	pxor	%xmm1, %xmm1

	pcmpeqb	(%esi), %xmm1
	add	%ecx, %ebx
	pmovmskb %xmm1, %edx
	shr	%cl, %edx
# ifdef USE_AS_STPCPY
	cmp	$16, %ebx
	jbe	L(CopyFrom1To16BytesTailCase2OrCase3)
# else
	cmp	$17, %ebx
	jbe	L(CopyFrom1To16BytesTailCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesTail)

	pcmpeqb	16(%esi), %xmm0
	pmovmskb %xmm0, %edx
# ifdef USE_AS_STPCPY
	cmp	$32, %ebx
	jbe	L(CopyFrom1To32BytesCase2OrCase3)
# else
	cmp	$33, %ebx
	jbe	L(CopyFrom1To32BytesCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyFrom1To32Bytes)

	movdqu	(%esi, %ecx), %xmm1   /* copy 16 bytes */
	movdqu	%xmm1, (%edi)

	sub	%ecx, %edi

/* If source address alignment != destination address alignment */
	.p2align 4
L(Unalign16Both):
	mov	$16, %ecx
	movdqa	(%esi, %ecx), %xmm1
	movaps	16(%esi, %ecx), %xmm2
	movdqu	%xmm1, (%edi, %ecx)
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %edx
	add	$16, %ecx
	sub	$48, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesUnalignedXmm2)

	movaps	16(%esi, %ecx), %xmm3
	movdqu	%xmm2, (%edi, %ecx)
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %edx
	add	$16, %ecx
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesUnalignedXmm3)

	movaps	16(%esi, %ecx), %xmm4
	movdqu	%xmm3, (%edi, %ecx)
	pcmpeqb	%xmm4, %xmm0
	pmovmskb %xmm0, %edx
	add	$16, %ecx
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesUnalignedXmm4)

	movaps	16(%esi, %ecx), %xmm1
	movdqu	%xmm4, (%edi, %ecx)
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %edx
	add	$16, %ecx
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesUnalignedXmm1)

	movaps	16(%esi, %ecx), %xmm2
	movdqu	%xmm1, (%edi, %ecx)
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %edx
	add	$16, %ecx
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesUnalignedXmm2)

	movaps	16(%esi, %ecx), %xmm3
	movdqu	%xmm2, (%edi, %ecx)
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %edx
	add	$16, %ecx
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesUnalignedXmm3)

	movdqu	%xmm3, (%edi, %ecx)
	mov	%esi, %edx
	lea	16(%esi, %ecx), %esi
	and	$-0x40, %esi
	sub	%esi, %edx
	sub	%edx, %edi
	lea	128(%ebx, %edx), %ebx

L(Unaligned64Loop):
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
	sub	$64, %ebx
	jbe	L(UnalignedLeaveCase2OrCase3)
	test	%edx, %edx
	jnz	L(Unaligned64Leave)
L(Unaligned64Loop_start):
	add	$64, %edi
	add	$64, %esi
	movdqu	%xmm4, -64(%edi)
	movaps	(%esi), %xmm2
	movdqa	%xmm2, %xmm4
	movdqu	%xmm5, -48(%edi)
	movaps	16(%esi), %xmm5
	pminub	%xmm5, %xmm2
	movaps	32(%esi), %xmm3
	movdqu	%xmm6, -32(%edi)
	movaps	%xmm3, %xmm6
	movdqu	%xmm7, -16(%edi)
	movaps	48(%esi), %xmm7
	pminub	%xmm7, %xmm3
	pminub	%xmm2, %xmm3
	pcmpeqb	%xmm0, %xmm3
	pmovmskb %xmm3, %edx
	sub	$64, %ebx
	jbe	L(UnalignedLeaveCase2OrCase3)
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
	movdqu	%xmm4, (%edi)
	movdqu	%xmm5, 16(%edi)
	movdqu	%xmm6, 32(%edi)
# ifdef USE_AS_STPCPY
	lea	48(%edi, %edx), %eax
# endif
	movdqu	%xmm7, 48(%edi)
	add	$15, %ebx
	sub	%edx, %ebx
	lea	49(%edi, %edx), %edi
	jmp	L(StrncpyFillTailWithZero)

/* If source address alignment == destination address alignment */

L(SourceStringAlignmentZero):
	pxor	%xmm0, %xmm0
	movdqa	(%esi), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %edx
# ifdef USE_AS_STPCPY
	cmp	$16, %ebx
	jbe	L(CopyFrom1To16BytesTail1Case2OrCase3)
# else
	cmp	$17, %ebx
	jbe	L(CopyFrom1To16BytesTail1Case2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesTail1)

	pcmpeqb	16(%esi), %xmm0
	movdqu	%xmm1, (%edi)
	pmovmskb %xmm0, %edx
# ifdef USE_AS_STPCPY
	cmp	$32, %ebx
	jbe	L(CopyFrom1To32Bytes1Case2OrCase3)
# else
	cmp	$33, %ebx
	jbe	L(CopyFrom1To32Bytes1Case2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyFrom1To32Bytes1)

	jmp	L(Unalign16Both)

/*-----------------End of main part---------------------------*/

/* Case1 */
	.p2align 4
L(CopyFrom1To16BytesTail):
	sub	%ecx, %ebx
	add	%ecx, %esi
	bsf	%edx, %edx
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %edx, 4)

	.p2align 4
L(CopyFrom1To32Bytes1):
	add	$16, %esi
	add	$16, %edi
	sub	$16, %ebx
L(CopyFrom1To16BytesTail1):
	bsf	%edx, %edx
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %edx, 4)

	.p2align 4
L(CopyFrom1To32Bytes):
	sub	%ecx, %ebx
	bsf	%edx, %edx
	add	%ecx, %esi
	add	$16, %edx
	sub	%ecx, %edx
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %edx, 4)

	.p2align 4
L(CopyFrom1To16BytesUnaligned_0):
	bsf	%edx, %edx
# ifdef USE_AS_STPCPY
	lea	(%edi, %edx), %eax
# endif
	movdqu	%xmm4, (%edi)
	add	$63, %ebx
	sub	%edx, %ebx
	lea	1(%edi, %edx), %edi
	jmp	L(StrncpyFillTailWithZero)

	.p2align 4
L(CopyFrom1To16BytesUnaligned_16):
	bsf	%ecx, %edx
	movdqu	%xmm4, (%edi)
# ifdef USE_AS_STPCPY
	lea	16(%edi, %edx), %eax
# endif
	movdqu	%xmm5, 16(%edi)
	add	$47, %ebx
	sub	%edx, %ebx
	lea	17(%edi, %edx), %edi
	jmp	L(StrncpyFillTailWithZero)

	.p2align 4
L(CopyFrom1To16BytesUnaligned_32):
	bsf	%edx, %edx
	movdqu	%xmm4, (%edi)
	movdqu	%xmm5, 16(%edi)
# ifdef USE_AS_STPCPY
	lea	32(%edi, %edx), %eax
# endif
	movdqu	%xmm6, 32(%edi)
	add	$31, %ebx
	sub	%edx, %ebx
	lea	33(%edi, %edx), %edi
	jmp	L(StrncpyFillTailWithZero)

	.p2align 4
L(CopyFrom1To16BytesUnalignedXmm6):
	movdqu	%xmm6, (%edi, %ecx)
	jmp	L(CopyFrom1To16BytesXmmExit)

	.p2align 4
L(CopyFrom1To16BytesUnalignedXmm5):
	movdqu	%xmm5, (%edi, %ecx)
	jmp	L(CopyFrom1To16BytesXmmExit)

	.p2align 4
L(CopyFrom1To16BytesUnalignedXmm4):
	movdqu	%xmm4, (%edi, %ecx)
	jmp	L(CopyFrom1To16BytesXmmExit)

	.p2align 4
L(CopyFrom1To16BytesUnalignedXmm3):
	movdqu	%xmm3, (%edi, %ecx)
	jmp	L(CopyFrom1To16BytesXmmExit)

	.p2align 4
L(CopyFrom1To16BytesUnalignedXmm1):
	movdqu	%xmm1, (%edi, %ecx)
	jmp	L(CopyFrom1To16BytesXmmExit)

	.p2align 4
L(CopyFrom1To16BytesExit):
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %edx, 4)

/* Case2 */

	.p2align 4
L(CopyFrom1To16BytesCase2):
	add	$16, %ebx
	add	%ecx, %edi
	add	%ecx, %esi
	bsf	%edx, %edx
	cmp	%ebx, %edx
	jb	L(CopyFrom1To16BytesExit)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %ebx, 4)

	.p2align 4
L(CopyFrom1To32BytesCase2):
	sub	%ecx, %ebx
	add	%ecx, %esi
	bsf	%edx, %edx
	add	$16, %edx
	sub	%ecx, %edx
	cmp	%ebx, %edx
	jb	L(CopyFrom1To16BytesExit)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %ebx, 4)

L(CopyFrom1To16BytesTailCase2):
	sub	%ecx, %ebx
	add	%ecx, %esi
	bsf	%edx, %edx
	cmp	%ebx, %edx
	jb	L(CopyFrom1To16BytesExit)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %ebx, 4)

L(CopyFrom1To16BytesTail1Case2):
	bsf	%edx, %edx
	cmp	%ebx, %edx
	jb	L(CopyFrom1To16BytesExit)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %ebx, 4)

/* Case2 or Case3,  Case3 */

	.p2align 4
L(CopyFrom1To16BytesCase2OrCase3):
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesCase2)
L(CopyFrom1To16BytesCase3):
	add	$16, %ebx
	add	%ecx, %edi
	add	%ecx, %esi
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %ebx, 4)

	.p2align 4
L(CopyFrom1To32BytesCase2OrCase3):
	test	%edx, %edx
	jnz	L(CopyFrom1To32BytesCase2)
	sub	%ecx, %ebx
	add	%ecx, %esi
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %ebx, 4)

	.p2align 4
L(CopyFrom1To16BytesTailCase2OrCase3):
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesTailCase2)
	sub	%ecx, %ebx
	add	%ecx, %esi
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %ebx, 4)

	.p2align 4
L(CopyFrom1To32Bytes1Case2OrCase3):
	add	$16, %edi
	add	$16, %esi
	sub	$16, %ebx
L(CopyFrom1To16BytesTail1Case2OrCase3):
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesTail1Case2)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %ebx, 4)

	.p2align 4
L(Exit0):
# ifdef USE_AS_STPCPY
	mov	%edi, %eax
# endif
	RETURN

	.p2align 4
L(Exit1):
	movb	%dh, (%edi)
# ifdef USE_AS_STPCPY
	lea	(%edi), %eax
# endif
	sub	$1, %ebx
	lea	1(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit2):
	movw	(%esi), %dx
	movw	%dx, (%edi)
# ifdef USE_AS_STPCPY
	lea	1(%edi), %eax
# endif
	sub	$2, %ebx
	lea	2(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit3):
	movw	(%esi), %cx
	movw	%cx, (%edi)
	movb	%dh, 2(%edi)
# ifdef USE_AS_STPCPY
	lea	2(%edi), %eax
# endif
	sub	$3, %ebx
	lea	3(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit4):
	movl	(%esi), %edx
	movl	%edx, (%edi)
# ifdef USE_AS_STPCPY
	lea	3(%edi), %eax
# endif
	sub	$4, %ebx
	lea	4(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit5):
	movl	(%esi), %ecx
	movb	%dh, 4(%edi)
	movl	%ecx, (%edi)
# ifdef USE_AS_STPCPY
	lea	4(%edi), %eax
# endif
	sub	$5, %ebx
	lea	5(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit6):
	movl	(%esi), %ecx
	movw	4(%esi), %dx
	movl	%ecx, (%edi)
	movw	%dx, 4(%edi)
# ifdef USE_AS_STPCPY
	lea	5(%edi), %eax
# endif
	sub	$6, %ebx
	lea	6(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit7):
	movl	(%esi), %ecx
	movl	3(%esi), %edx
	movl	%ecx, (%edi)
	movl	%edx, 3(%edi)
# ifdef USE_AS_STPCPY
	lea	6(%edi), %eax
# endif
	sub	$7, %ebx
	lea	7(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit8):
	movlpd	(%esi), %xmm0
	movlpd	%xmm0, (%edi)
# ifdef USE_AS_STPCPY
	lea	7(%edi), %eax
# endif
	sub	$8, %ebx
	lea	8(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit9):
	movlpd	(%esi), %xmm0
	movb	%dh, 8(%edi)
	movlpd	%xmm0, (%edi)
# ifdef USE_AS_STPCPY
	lea	8(%edi), %eax
# endif
	sub	$9, %ebx
	lea	9(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit10):
	movlpd	(%esi), %xmm0
	movw	8(%esi), %dx
	movlpd	%xmm0, (%edi)
	movw	%dx, 8(%edi)
# ifdef USE_AS_STPCPY
	lea	9(%edi), %eax
# endif
	sub	$10, %ebx
	lea	10(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit11):
	movlpd	(%esi), %xmm0
	movl	7(%esi), %edx
	movlpd	%xmm0, (%edi)
	movl	%edx, 7(%edi)
# ifdef USE_AS_STPCPY
	lea	10(%edi), %eax
# endif
	sub	$11, %ebx
	lea	11(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit12):
	movlpd	(%esi), %xmm0
	movl	8(%esi), %edx
	movlpd	%xmm0, (%edi)
	movl	%edx, 8(%edi)
# ifdef USE_AS_STPCPY
	lea	11(%edi), %eax
# endif
	sub	$12, %ebx
	lea	12(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit13):
	movlpd	(%esi), %xmm0
	movlpd	5(%esi), %xmm1
	movlpd	%xmm0, (%edi)
	movlpd	%xmm1, 5(%edi)
# ifdef USE_AS_STPCPY
	lea	12(%edi), %eax
# endif
	sub	$13, %ebx
	lea	13(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit14):
	movlpd	(%esi), %xmm0
	movlpd	6(%esi), %xmm1
	movlpd	%xmm0, (%edi)
	movlpd	%xmm1, 6(%edi)
# ifdef USE_AS_STPCPY
	lea	13(%edi), %eax
# endif
	sub	$14, %ebx
	lea	14(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit15):
	movlpd	(%esi), %xmm0
	movlpd	7(%esi), %xmm1
	movlpd	%xmm0, (%edi)
	movlpd	%xmm1, 7(%edi)
# ifdef USE_AS_STPCPY
	lea	14(%edi), %eax
# endif
	sub	$15, %ebx
	lea	15(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit16):
	movdqu	(%esi), %xmm0
	movdqu	%xmm0, (%edi)
# ifdef USE_AS_STPCPY
	lea	15(%edi), %eax
# endif
	sub	$16, %ebx
	lea	16(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit17):
	movdqu	(%esi), %xmm0
	movdqu	%xmm0, (%edi)
	movb	%dh, 16(%edi)
# ifdef USE_AS_STPCPY
	lea	16(%edi), %eax
# endif
	sub	$17, %ebx
	lea	17(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit18):
	movdqu	(%esi), %xmm0
	movw	16(%esi), %cx
	movdqu	%xmm0, (%edi)
	movw	%cx, 16(%edi)
# ifdef USE_AS_STPCPY
	lea	17(%edi), %eax
# endif
	sub	$18, %ebx
	lea	18(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit19):
	movdqu	(%esi), %xmm0
	movl	15(%esi), %ecx
	movdqu	%xmm0, (%edi)
	movl	%ecx, 15(%edi)
# ifdef USE_AS_STPCPY
	lea	18(%edi), %eax
# endif
	sub	$19, %ebx
	lea	19(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit20):
	movdqu	(%esi), %xmm0
	movl	16(%esi), %ecx
	movdqu	%xmm0, (%edi)
	movl	%ecx, 16(%edi)
# ifdef USE_AS_STPCPY
	lea	19(%edi), %eax
# endif
	sub	$20, %ebx
	lea	20(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit21):
	movdqu	(%esi), %xmm0
	movl	16(%esi), %ecx
	movdqu	%xmm0, (%edi)
	movl	%ecx, 16(%edi)
	movb	%dh, 20(%edi)
# ifdef USE_AS_STPCPY
	lea	20(%edi), %eax
# endif
	sub	$21, %ebx
	lea	21(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit22):
	movdqu	(%esi), %xmm0
	movlpd	14(%esi), %xmm3
	movdqu	%xmm0, (%edi)
	movlpd	%xmm3, 14(%edi)
# ifdef USE_AS_STPCPY
	lea	21(%edi), %eax
# endif
	sub	$22, %ebx
	lea	22(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit23):
	movdqu	(%esi), %xmm0
	movlpd	15(%esi), %xmm3
	movdqu	%xmm0, (%edi)
	movlpd	%xmm3, 15(%edi)
# ifdef USE_AS_STPCPY
	lea	22(%edi), %eax
# endif
	sub	$23, %ebx
	lea	23(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit24):
	movdqu	(%esi), %xmm0
	movlpd	16(%esi), %xmm2
	movdqu	%xmm0, (%edi)
	movlpd	%xmm2, 16(%edi)
# ifdef USE_AS_STPCPY
	lea	23(%edi), %eax
# endif
	sub	$24, %ebx
	lea	24(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit25):
	movdqu	(%esi), %xmm0
	movlpd	16(%esi), %xmm2
	movdqu	%xmm0, (%edi)
	movlpd	%xmm2, 16(%edi)
	movb	%dh, 24(%edi)
# ifdef USE_AS_STPCPY
	lea	24(%edi), %eax
# endif
	sub	$25, %ebx
	lea	25(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit26):
	movdqu	(%esi), %xmm0
	movlpd	16(%esi), %xmm2
	movw	24(%esi), %cx
	movdqu	%xmm0, (%edi)
	movlpd	%xmm2, 16(%edi)
	movw	%cx, 24(%edi)
# ifdef USE_AS_STPCPY
	lea	25(%edi), %eax
# endif
	sub	$26, %ebx
	lea	26(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit27):
	movdqu	(%esi), %xmm0
	movlpd	16(%esi), %xmm2
	movl	23(%esi), %ecx
	movdqu	%xmm0, (%edi)
	movlpd	%xmm2, 16(%edi)
	movl	%ecx, 23(%edi)
# ifdef USE_AS_STPCPY
	lea	26(%edi), %eax
# endif
	sub	$27, %ebx
	lea	27(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit28):
	movdqu	(%esi), %xmm0
	movlpd	16(%esi), %xmm2
	movl	24(%esi), %ecx
	movdqu	%xmm0, (%edi)
	movlpd	%xmm2, 16(%edi)
	movl	%ecx, 24(%edi)
# ifdef USE_AS_STPCPY
	lea	27(%edi), %eax
# endif
	sub	$28, %ebx
	lea	28(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit29):
	movdqu	(%esi), %xmm0
	movdqu	13(%esi), %xmm2
	movdqu	%xmm0, (%edi)
	movdqu	%xmm2, 13(%edi)
# ifdef USE_AS_STPCPY
	lea	28(%edi), %eax
# endif
	sub	$29, %ebx
	lea	29(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit30):
	movdqu	(%esi), %xmm0
	movdqu	14(%esi), %xmm2
	movdqu	%xmm0, (%edi)
	movdqu	%xmm2, 14(%edi)
# ifdef USE_AS_STPCPY
	lea	29(%edi), %eax
# endif
	sub	$30, %ebx
	lea	30(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN


	.p2align 4
L(Exit31):
	movdqu	(%esi), %xmm0
	movdqu	15(%esi), %xmm2
	movdqu	%xmm0, (%edi)
	movdqu	%xmm2, 15(%edi)
# ifdef USE_AS_STPCPY
	lea	30(%edi), %eax
# endif
	sub	$31, %ebx
	lea	31(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(Exit32):
	movdqu	(%esi), %xmm0
	movdqu	16(%esi), %xmm2
	movdqu	%xmm0, (%edi)
	movdqu	%xmm2, 16(%edi)
# ifdef USE_AS_STPCPY
	lea	31(%edi), %eax
# endif
	sub	$32, %ebx
	lea	32(%edi), %edi
	jnz	L(StrncpyFillTailWithZero)
	RETURN

	.p2align 4
L(StrncpyExit1):
	movb	(%esi), %dl
	movb	%dl, (%edi)
# ifdef USE_AS_STPCPY
	lea	1(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit2):
	movw	(%esi), %dx
	movw	%dx, (%edi)
# ifdef USE_AS_STPCPY
	lea	2(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit3):
	movw	(%esi), %cx
	movb	2(%esi), %dl
	movw	%cx, (%edi)
	movb	%dl, 2(%edi)
# ifdef USE_AS_STPCPY
	lea	3(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit4):
	movl	(%esi), %edx
	movl	%edx, (%edi)
# ifdef USE_AS_STPCPY
	lea	4(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit5):
	movl	(%esi), %ecx
	movb	4(%esi), %dl
	movl	%ecx, (%edi)
	movb	%dl, 4(%edi)
# ifdef USE_AS_STPCPY
	lea	5(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit6):
	movl	(%esi), %ecx
	movw	4(%esi), %dx
	movl	%ecx, (%edi)
	movw	%dx, 4(%edi)
# ifdef USE_AS_STPCPY
	lea	6(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit7):
	movl	(%esi), %ecx
	movl	3(%esi), %edx
	movl	%ecx, (%edi)
	movl	%edx, 3(%edi)
# ifdef USE_AS_STPCPY
	lea	7(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit8):
	movlpd	(%esi), %xmm0
	movlpd	%xmm0, (%edi)
# ifdef USE_AS_STPCPY
	lea	8(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit9):
	movlpd	(%esi), %xmm0
	movb	8(%esi), %dl
	movlpd	%xmm0, (%edi)
	movb	%dl, 8(%edi)
# ifdef USE_AS_STPCPY
	lea	9(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit10):
	movlpd	(%esi), %xmm0
	movw	8(%esi), %dx
	movlpd	%xmm0, (%edi)
	movw	%dx, 8(%edi)
# ifdef USE_AS_STPCPY
	lea	10(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit11):
	movlpd	(%esi), %xmm0
	movl	7(%esi), %edx
	movlpd	%xmm0, (%edi)
	movl	%edx, 7(%edi)
# ifdef USE_AS_STPCPY
	lea	11(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit12):
	movlpd	(%esi), %xmm0
	movl	8(%esi), %edx
	movlpd	%xmm0, (%edi)
	movl	%edx, 8(%edi)
# ifdef USE_AS_STPCPY
	lea	12(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit13):
	movlpd	(%esi), %xmm0
	movlpd	5(%esi), %xmm1
	movlpd	%xmm0, (%edi)
	movlpd	%xmm1, 5(%edi)
# ifdef USE_AS_STPCPY
	lea	13(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit14):
	movlpd	(%esi), %xmm0
	movlpd	6(%esi), %xmm1
	movlpd	%xmm0, (%edi)
	movlpd	%xmm1, 6(%edi)
# ifdef USE_AS_STPCPY
	lea	14(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit15):
	movlpd	(%esi), %xmm0
	movlpd	7(%esi), %xmm1
	movlpd	%xmm0, (%edi)
	movlpd	%xmm1, 7(%edi)
# ifdef USE_AS_STPCPY
	lea	15(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit16):
	movdqu	(%esi), %xmm0
	movdqu	%xmm0, (%edi)
# ifdef USE_AS_STPCPY
	lea	16(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit17):
	movdqu	(%esi), %xmm0
	movb	16(%esi), %cl
	movdqu	%xmm0, (%edi)
	movb	%cl, 16(%edi)
# ifdef USE_AS_STPCPY
	lea	17(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit18):
	movdqu	(%esi), %xmm0
	movw	16(%esi), %cx
	movdqu	%xmm0, (%edi)
	movw	%cx, 16(%edi)
# ifdef USE_AS_STPCPY
	lea	18(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit19):
	movdqu	(%esi), %xmm0
	movl	15(%esi), %ecx
	movdqu	%xmm0, (%edi)
	movl	%ecx, 15(%edi)
# ifdef USE_AS_STPCPY
	lea	19(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit20):
	movdqu	(%esi), %xmm0
	movl	16(%esi), %ecx
	movdqu	%xmm0, (%edi)
	movl	%ecx, 16(%edi)
# ifdef USE_AS_STPCPY
	lea	20(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit21):
	movdqu	(%esi), %xmm0
	movl	16(%esi), %ecx
	movb	20(%esi), %dl
	movdqu	%xmm0, (%edi)
	movl	%ecx, 16(%edi)
	movb	%dl, 20(%edi)
# ifdef USE_AS_STPCPY
	lea	21(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit22):
	movdqu	(%esi), %xmm0
	movlpd	14(%esi), %xmm3
	movdqu	%xmm0, (%edi)
	movlpd	%xmm3, 14(%edi)
# ifdef USE_AS_STPCPY
	lea	22(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit23):
	movdqu	(%esi), %xmm0
	movlpd	15(%esi), %xmm3
	movdqu	%xmm0, (%edi)
	movlpd	%xmm3, 15(%edi)
# ifdef USE_AS_STPCPY
	lea	23(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit24):
	movdqu	(%esi), %xmm0
	movlpd	16(%esi), %xmm2
	movdqu	%xmm0, (%edi)
	movlpd	%xmm2, 16(%edi)
# ifdef USE_AS_STPCPY
	lea	24(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit25):
	movdqu	(%esi), %xmm0
	movlpd	16(%esi), %xmm2
	movb	24(%esi), %cl
	movdqu	%xmm0, (%edi)
	movlpd	%xmm2, 16(%edi)
	movb	%cl, 24(%edi)
# ifdef USE_AS_STPCPY
	lea	25(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit26):
	movdqu	(%esi), %xmm0
	movlpd	16(%esi), %xmm2
	movw	24(%esi), %cx
	movdqu	%xmm0, (%edi)
	movlpd	%xmm2, 16(%edi)
	movw	%cx, 24(%edi)
# ifdef USE_AS_STPCPY
	lea	26(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit27):
	movdqu	(%esi), %xmm0
	movlpd	16(%esi), %xmm2
	movl	23(%esi), %ecx
	movdqu	%xmm0, (%edi)
	movlpd	%xmm2, 16(%edi)
	movl	%ecx, 23(%edi)
# ifdef USE_AS_STPCPY
	lea	27(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit28):
	movdqu	(%esi), %xmm0
	movlpd	16(%esi), %xmm2
	movl	24(%esi), %ecx
	movdqu	%xmm0, (%edi)
	movlpd	%xmm2, 16(%edi)
	movl	%ecx, 24(%edi)
# ifdef USE_AS_STPCPY
	lea	28(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit29):
	movdqu	(%esi), %xmm0
	movdqu	13(%esi), %xmm2
	movdqu	%xmm0, (%edi)
	movdqu	%xmm2, 13(%edi)
# ifdef USE_AS_STPCPY
	lea	29(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit30):
	movdqu	(%esi), %xmm0
	movdqu	14(%esi), %xmm2
	movdqu	%xmm0, (%edi)
	movdqu	%xmm2, 14(%edi)
# ifdef USE_AS_STPCPY
	lea	30(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit31):
	movdqu	(%esi), %xmm0
	movdqu	15(%esi), %xmm2
	movdqu	%xmm0, (%edi)
	movdqu	%xmm2, 15(%edi)
# ifdef USE_AS_STPCPY
	lea	31(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit32):
	movdqu	(%esi), %xmm0
	movdqu	16(%esi), %xmm2
	movdqu	%xmm0, (%edi)
	movdqu	%xmm2, 16(%edi)
# ifdef USE_AS_STPCPY
	lea	32(%edi), %eax
# endif
	RETURN

	.p2align 4
L(StrncpyExit33):
	movdqu	(%esi), %xmm0
	movdqu	16(%esi), %xmm2
	movb	32(%esi), %cl
	movdqu	%xmm0, (%edi)
	movdqu	%xmm2, 16(%edi)
	movb	%cl, 32(%edi)
	RETURN

	.p2align 4
L(Fill0):
	RETURN

	.p2align 4
L(Fill1):
	movb	%dl, (%edi)
	RETURN

	.p2align 4
L(Fill2):
	movw	%dx, (%edi)
	RETURN

	.p2align 4
L(Fill3):
	movl	%edx, -1(%edi)
	RETURN

	.p2align 4
L(Fill4):
	movl	%edx, (%edi)
	RETURN

	.p2align 4
L(Fill5):
	movl	%edx, (%edi)
	movb	%dl, 4(%edi)
	RETURN

	.p2align 4
L(Fill6):
	movl	%edx, (%edi)
	movw	%dx, 4(%edi)
	RETURN

	.p2align 4
L(Fill7):
	movlpd	%xmm0, -1(%edi)
	RETURN

	.p2align 4
L(Fill8):
	movlpd	%xmm0, (%edi)
	RETURN

	.p2align 4
L(Fill9):
	movlpd	%xmm0, (%edi)
	movb	%dl, 8(%edi)
	RETURN

	.p2align 4
L(Fill10):
	movlpd	%xmm0, (%edi)
	movw	%dx, 8(%edi)
	RETURN

	.p2align 4
L(Fill11):
	movlpd	%xmm0, (%edi)
	movl	%edx, 7(%edi)
	RETURN

	.p2align 4
L(Fill12):
	movlpd	%xmm0, (%edi)
	movl	%edx, 8(%edi)
	RETURN

	.p2align 4
L(Fill13):
	movlpd	%xmm0, (%edi)
	movlpd	%xmm0, 5(%edi)
	RETURN

	.p2align 4
L(Fill14):
	movlpd	%xmm0, (%edi)
	movlpd	%xmm0, 6(%edi)
	RETURN

	.p2align 4
L(Fill15):
	movdqu	%xmm0, -1(%edi)
	RETURN

	.p2align 4
L(Fill16):
	movdqu	%xmm0, (%edi)
	RETURN

	.p2align 4
L(CopyFrom1To16BytesUnalignedXmm2):
	movdqu	%xmm2, (%edi, %ecx)

	.p2align 4
L(CopyFrom1To16BytesXmmExit):
	bsf	%edx, %edx
	add	$15, %ebx
	add	%ecx, %edi
# ifdef USE_AS_STPCPY
	lea	(%edi, %edx), %eax
# endif
	sub	%edx, %ebx
	lea	1(%edi, %edx), %edi

	.p2align 4
L(StrncpyFillTailWithZero):
	pxor	%xmm0, %xmm0
	xor	%edx, %edx
	sub	$16, %ebx
	jbe	L(StrncpyFillExit)

	movdqu	%xmm0, (%edi)
	add	$16, %edi

	mov	%edi, %esi
	and	$0xf, %esi
	sub	%esi, %edi
	add	%esi, %ebx
	sub	$64, %ebx
	jb	L(StrncpyFillLess64)

L(StrncpyFillLoopMovdqa):
	movdqa	%xmm0, (%edi)
	movdqa	%xmm0, 16(%edi)
	movdqa	%xmm0, 32(%edi)
	movdqa	%xmm0, 48(%edi)
	add	$64, %edi
	sub	$64, %ebx
	jae	L(StrncpyFillLoopMovdqa)

L(StrncpyFillLess64):
	add	$32, %ebx
	jl	L(StrncpyFillLess32)
	movdqa	%xmm0, (%edi)
	movdqa	%xmm0, 16(%edi)
	add	$32, %edi
	sub	$16, %ebx
	jl	L(StrncpyFillExit)
	movdqa	%xmm0, (%edi)
	add	$16, %edi
	BRANCH_TO_JMPTBL_ENTRY (L(FillTable), %ebx, 4)

L(StrncpyFillLess32):
	add	$16, %ebx
	jl	L(StrncpyFillExit)
	movdqa	%xmm0, (%edi)
	add	$16, %edi
	BRANCH_TO_JMPTBL_ENTRY (L(FillTable), %ebx, 4)

L(StrncpyFillExit):
	add	$16, %ebx
	BRANCH_TO_JMPTBL_ENTRY (L(FillTable), %ebx, 4)

	.p2align 4
L(UnalignedLeaveCase2OrCase3):
	test	%edx, %edx
	jnz	L(Unaligned64LeaveCase2)
L(Unaligned64LeaveCase3):
	lea	64(%ebx), %ecx
	and	$-16, %ecx
	add	$48, %ebx
	jl	L(CopyFrom1To16BytesCase3)
	movdqu	%xmm4, (%edi)
	sub	$16, %ebx
	jb	L(CopyFrom1To16BytesCase3)
	movdqu	%xmm5, 16(%edi)
	sub	$16, %ebx
	jb	L(CopyFrom1To16BytesCase3)
	movdqu	%xmm6, 32(%edi)
	sub	$16, %ebx
	jb	L(CopyFrom1To16BytesCase3)
	movdqu	%xmm7, 48(%edi)
# ifdef USE_AS_STPCPY
	lea	64(%edi), %eax
# endif
	RETURN

	.p2align 4
L(Unaligned64LeaveCase2):
	xor	%ecx, %ecx
	pcmpeqb	%xmm4, %xmm0
	pmovmskb %xmm0, %edx
	add	$48, %ebx
	jle	L(CopyFrom1To16BytesCase2OrCase3)
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesUnalignedXmm4)

	pcmpeqb	%xmm5, %xmm0
	pmovmskb %xmm0, %edx
	movdqu	%xmm4, (%edi)
	add	$16, %ecx
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesUnalignedXmm5)

	pcmpeqb	%xmm6, %xmm0
	pmovmskb %xmm0, %edx
	movdqu	%xmm5, 16(%edi)
	add	$16, %ecx
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%edx, %edx
	jnz	L(CopyFrom1To16BytesUnalignedXmm6)

	pcmpeqb	%xmm7, %xmm0
	pmovmskb %xmm0, %edx
	movdqu	%xmm6, 32(%edi)
	lea	16(%edi, %ecx), %edi
	lea	16(%esi, %ecx), %esi
	bsf	%edx, %edx
	cmp	%ebx, %edx
	jb	L(CopyFrom1To16BytesExit)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %ebx, 4)

	.p2align 4
L(ExitZero):
	movl	%edi, %eax
	RETURN

END (STRCPY)

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
	.int    JMPTBL(L(Exit23), L(ExitTable))
	.int	JMPTBL(L(Exit24), L(ExitTable))
	.int	JMPTBL(L(Exit25), L(ExitTable))
	.int	JMPTBL(L(Exit26), L(ExitTable))
	.int	JMPTBL(L(Exit27), L(ExitTable))
	.int	JMPTBL(L(Exit28), L(ExitTable))
	.int	JMPTBL(L(Exit29), L(ExitTable))
	.int	JMPTBL(L(Exit30), L(ExitTable))
	.int	JMPTBL(L(Exit31), L(ExitTable))
	.int	JMPTBL(L(Exit32), L(ExitTable))

L(ExitStrncpyTable):
	.int	JMPTBL(L(Exit0), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit1), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit2), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit3), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit4), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit5), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit6), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit7), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit8), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit9), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit10), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit11), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit12), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit13), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit14), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit15), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit16), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit17), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit18), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit19), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit20), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit21), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit22), L(ExitStrncpyTable))
	.int    JMPTBL(L(StrncpyExit23), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit24), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit25), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit26), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit27), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit28), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit29), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit30), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit31), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit32), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit33), L(ExitStrncpyTable))

	.p2align 4
L(FillTable):
	.int	JMPTBL(L(Fill0), L(FillTable))
	.int	JMPTBL(L(Fill1), L(FillTable))
	.int	JMPTBL(L(Fill2), L(FillTable))
	.int	JMPTBL(L(Fill3), L(FillTable))
	.int	JMPTBL(L(Fill4), L(FillTable))
	.int	JMPTBL(L(Fill5), L(FillTable))
	.int	JMPTBL(L(Fill6), L(FillTable))
	.int	JMPTBL(L(Fill7), L(FillTable))
	.int	JMPTBL(L(Fill8), L(FillTable))
	.int	JMPTBL(L(Fill9), L(FillTable))
	.int	JMPTBL(L(Fill10), L(FillTable))
	.int	JMPTBL(L(Fill11), L(FillTable))
	.int	JMPTBL(L(Fill12), L(FillTable))
	.int	JMPTBL(L(Fill13), L(FillTable))
	.int	JMPTBL(L(Fill14), L(FillTable))
	.int	JMPTBL(L(Fill15), L(FillTable))
	.int	JMPTBL(L(Fill16), L(FillTable))
# else
#  define PARMS  4
#  define ENTRANCE
#  define RETURN  POP (%edi); ret; CFI_PUSH (%edi)
#  define RETURN1  ret

	.text
ENTRY (STRCPY)
	ENTRANCE
	mov	STR1(%esp), %edx
	mov	STR2(%esp), %ecx

	cmpb	$0, (%ecx)
	jz	L(ExitTail1)
	cmpb	$0, 1(%ecx)
	jz	L(ExitTail2)
	cmpb	$0, 2(%ecx)
	jz	L(ExitTail3)
	cmpb	$0, 3(%ecx)
	jz	L(ExitTail4)
	cmpb	$0, 4(%ecx)
	jz	L(ExitTail5)
	cmpb	$0, 5(%ecx)
	jz	L(ExitTail6)
	cmpb	$0, 6(%ecx)
	jz	L(ExitTail7)
	cmpb	$0, 7(%ecx)
	jz	L(ExitTail8)
	cmpb	$0, 8(%ecx)
	jz	L(ExitTail9)
	cmpb	$0, 9(%ecx)
	jz	L(ExitTail10)
	cmpb	$0, 10(%ecx)
	jz	L(ExitTail11)
	cmpb	$0, 11(%ecx)
	jz	L(ExitTail12)
	cmpb	$0, 12(%ecx)
	jz	L(ExitTail13)
	cmpb	$0, 13(%ecx)
	jz	L(ExitTail14)
	cmpb	$0, 14(%ecx)
	jz	L(ExitTail15)
	cmpb	$0, 15(%ecx)
	jz	L(ExitTail16)

	PUSH	(%edi)
	PUSH	(%ebx)

	mov	%edx, %edi
	lea	16(%ecx), %ebx
	and	$-16, %ebx
	pxor	%xmm0, %xmm0
	movdqu	(%ecx), %xmm1
	movdqu	%xmm1, (%edx)
	pcmpeqb	(%ebx), %xmm0
	pmovmskb %xmm0, %eax
	sub	%ecx, %ebx
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	mov	%ecx, %eax
	lea	16(%ecx), %ecx
	and	$-16, %ecx
	sub	%ecx, %eax
	sub	%eax, %edx
	xor	%ebx, %ebx

	.p2align 4
	movdqa	(%ecx), %xmm1
	movaps	16(%ecx), %xmm2
	movdqu	%xmm1, (%edx)
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	add	$16, %ebx
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%ecx, %ebx), %xmm3
	movdqu	%xmm2, (%edx, %ebx)
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %eax
	add	$16, %ebx
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%ecx, %ebx), %xmm4
	movdqu	%xmm3, (%edx, %ebx)
	pcmpeqb	%xmm4, %xmm0
	pmovmskb %xmm0, %eax
	add	$16, %ebx
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%ecx, %ebx), %xmm1
	movdqu	%xmm4, (%edx, %ebx)
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	add	$16, %ebx
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%ecx, %ebx), %xmm2
	movdqu	%xmm1, (%edx, %ebx)
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	add	$16, %ebx
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%ecx, %ebx), %xmm3
	movdqu	%xmm2, (%edx, %ebx)
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %eax
	add	$16, %ebx
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movdqu	%xmm3, (%edx, %ebx)
	mov	%ecx, %eax
	lea	16(%ecx, %ebx), %ecx
	and	$-0x40, %ecx
	sub	%ecx, %eax
	sub	%eax, %edx

L(Aligned64Loop):
	movaps	(%ecx), %xmm2
	movaps	%xmm2, %xmm4
	movaps	16(%ecx), %xmm5
	movaps	32(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	48(%ecx), %xmm7
	pminub	%xmm5, %xmm2
	add	$64, %ecx
	pminub	%xmm7, %xmm3
	add	$64, %edx
	pminub	%xmm2, %xmm3
	pcmpeqb	%xmm0, %xmm3
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(Aligned64Leave)
L(Aligned64Loop_start):
	movdqu	%xmm4, -64(%edx)
	movaps	(%ecx), %xmm2
	movdqa	%xmm2, %xmm4
	movdqu	%xmm5, -48(%edx)
	movaps	16(%ecx), %xmm5
	pminub	%xmm5, %xmm2
	movaps	32(%ecx), %xmm3
	movdqu	%xmm6, -32(%edx)
	movaps	%xmm3, %xmm6
	movdqu	%xmm7, -16(%edx)
	movaps	48(%ecx), %xmm7
	pminub	%xmm7, %xmm3
	pminub	%xmm2, %xmm3
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %eax
	add	$64, %edx
	add	$64, %ecx
	test	%eax, %eax
	jz	L(Aligned64Loop_start)
L(Aligned64Leave):
	sub	$0xa0, %ebx
	pxor	%xmm0, %xmm0
	pcmpeqb	%xmm4, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqb	%xmm5, %xmm0
	pmovmskb %xmm0, %eax
	movdqu	%xmm4, -64(%edx)
	test	%eax, %eax
	lea	16(%ebx), %ebx
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqb	%xmm6, %xmm0
	pmovmskb %xmm0, %eax
	movdqu	%xmm5, -48(%edx)
	test	%eax, %eax
	lea	16(%ebx), %ebx
	jnz	L(CopyFrom1To16Bytes)

	movdqu	%xmm6, -32(%edx)
	pcmpeqb	%xmm7, %xmm0
	pmovmskb %xmm0, %eax
	lea	16(%ebx), %ebx

/*-----------------End of main part---------------------------*/

	.p2align 4
L(CopyFrom1To16Bytes):
	add	%ebx, %edx
	add	%ebx, %ecx

	POP	(%ebx)
	test	%al, %al
	jz	L(ExitHigh)
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
	/* Exit 8 */
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	4(%ecx), %eax
	movl	%eax, 4(%edx)
# ifdef USE_AS_STPCPY
	lea	7(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

	.p2align 4
L(ExitHigh):
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
	/* Exit 16 */
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	movlpd	8(%ecx), %xmm0
	movlpd	%xmm0, 8(%edx)
# ifdef USE_AS_STPCPY
	lea	15(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

	.p2align 4
L(Exit1):
	movb	(%ecx), %al
	movb	%al, (%edx)
# ifdef USE_AS_STPCPY
	lea	(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

	.p2align 4
L(Exit2):
	movw	(%ecx), %ax
	movw	%ax, (%edx)
# ifdef USE_AS_STPCPY
	lea	1(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

	.p2align 4
L(Exit3):
	movw	(%ecx), %ax
	movw	%ax, (%edx)
	movb	2(%ecx), %al
	movb	%al, 2(%edx)
# ifdef USE_AS_STPCPY
	lea	2(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

	.p2align 4
L(Exit4):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
# ifdef USE_AS_STPCPY
	lea	3(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

	.p2align 4
L(Exit5):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movb	4(%ecx), %al
	movb	%al, 4(%edx)
# ifdef USE_AS_STPCPY
	lea	4(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

	.p2align 4
L(Exit6):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movw	4(%ecx), %ax
	movw	%ax, 4(%edx)
# ifdef USE_AS_STPCPY
	lea	5(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

	.p2align 4
L(Exit7):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	3(%ecx), %eax
	movl	%eax, 3(%edx)
# ifdef USE_AS_STPCPY
	lea	6(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

	.p2align 4
L(Exit9):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	4(%ecx), %eax
	movl	%eax, 4(%edx)
	movb	8(%ecx), %al
	movb	%al, 8(%edx)
# ifdef USE_AS_STPCPY
	lea	8(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

	.p2align 4
L(Exit10):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	4(%ecx), %eax
	movl	%eax, 4(%edx)
	movw	8(%ecx), %ax
	movw	%ax, 8(%edx)
# ifdef USE_AS_STPCPY
	lea	9(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

	.p2align 4
L(Exit11):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	4(%ecx), %eax
	movl	%eax, 4(%edx)
	movl	7(%ecx), %eax
	movl	%eax, 7(%edx)
# ifdef USE_AS_STPCPY
	lea	10(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

	.p2align 4
L(Exit12):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	4(%ecx), %eax
	movl	%eax, 4(%edx)
	movl	8(%ecx), %eax
	movl	%eax, 8(%edx)
# ifdef USE_AS_STPCPY
	lea	11(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

	.p2align 4
L(Exit13):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	movlpd	5(%ecx), %xmm0
	movlpd	%xmm0, 5(%edx)
# ifdef USE_AS_STPCPY
	lea	12(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

	.p2align 4
L(Exit14):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	movlpd	6(%ecx), %xmm0
	movlpd	%xmm0, 6(%edx)
# ifdef USE_AS_STPCPY
	lea	13(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

	.p2align 4
L(Exit15):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	movlpd	7(%ecx), %xmm0
	movlpd	%xmm0, 7(%edx)
# ifdef USE_AS_STPCPY
	lea	14(%edx), %eax
# else
	movl	%edi, %eax
# endif
	RETURN

CFI_POP (%edi)

	.p2align 4
L(ExitTail1):
	movb	(%ecx), %al
	movb	%al, (%edx)
	movl	%edx, %eax
	RETURN1

	.p2align 4
L(ExitTail2):
	movw	(%ecx), %ax
	movw	%ax, (%edx)
# ifdef USE_AS_STPCPY
	lea	1(%edx), %eax
# else
	movl	%edx, %eax
# endif
	RETURN1

	.p2align 4
L(ExitTail3):
	movw	(%ecx), %ax
	movw	%ax, (%edx)
	movb	2(%ecx), %al
	movb	%al, 2(%edx)
# ifdef USE_AS_STPCPY
	lea	2(%edx), %eax
# else
	movl	%edx, %eax
# endif
	RETURN1

	.p2align 4
L(ExitTail4):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
# ifdef USE_AS_STPCPY
	lea	3(%edx), %eax
# else
	movl	%edx, %eax
# endif
	RETURN1

	.p2align 4
L(ExitTail5):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movb	4(%ecx), %al
	movb	%al, 4(%edx)
# ifdef USE_AS_STPCPY
	lea	4(%edx), %eax
# else
	movl	%edx, %eax
# endif
	RETURN1

	.p2align 4
L(ExitTail6):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movw	4(%ecx), %ax
	movw	%ax, 4(%edx)
# ifdef USE_AS_STPCPY
	lea	5(%edx), %eax
# else
	movl	%edx, %eax
# endif
	RETURN1

	.p2align 4
L(ExitTail7):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	3(%ecx), %eax
	movl	%eax, 3(%edx)
# ifdef USE_AS_STPCPY
	lea	6(%edx), %eax
# else
	movl	%edx, %eax
# endif
	RETURN1

	.p2align 4
L(ExitTail8):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	4(%ecx), %eax
	movl	%eax, 4(%edx)
# ifdef USE_AS_STPCPY
	lea	7(%edx), %eax
# else
	movl	%edx, %eax
# endif
	RETURN1

	.p2align 4
L(ExitTail9):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	4(%ecx), %eax
	movl	%eax, 4(%edx)
	movb	8(%ecx), %al
	movb	%al, 8(%edx)
# ifdef USE_AS_STPCPY
	lea	8(%edx), %eax
# else
	movl	%edx, %eax
# endif
	RETURN1

	.p2align 4
L(ExitTail10):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	4(%ecx), %eax
	movl	%eax, 4(%edx)
	movw	8(%ecx), %ax
	movw	%ax, 8(%edx)
# ifdef USE_AS_STPCPY
	lea	9(%edx), %eax
# else
	movl	%edx, %eax
# endif
	RETURN1

	.p2align 4
L(ExitTail11):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	4(%ecx), %eax
	movl	%eax, 4(%edx)
	movl	7(%ecx), %eax
	movl	%eax, 7(%edx)
# ifdef USE_AS_STPCPY
	lea	10(%edx), %eax
# else
	movl	%edx, %eax
# endif
	RETURN1

	.p2align 4
L(ExitTail12):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	4(%ecx), %eax
	movl	%eax, 4(%edx)
	movl	8(%ecx), %eax
	movl	%eax, 8(%edx)
# ifdef USE_AS_STPCPY
	lea	11(%edx), %eax
# else
	movl	%edx, %eax
# endif
	RETURN1

	.p2align 4
L(ExitTail13):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	movlpd	5(%ecx), %xmm0
	movlpd	%xmm0, 5(%edx)
# ifdef USE_AS_STPCPY
	lea	12(%edx), %eax
# else
	movl	%edx, %eax
# endif
	RETURN1

	.p2align 4
L(ExitTail14):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	movlpd	6(%ecx), %xmm0
	movlpd	%xmm0, 6(%edx)
# ifdef USE_AS_STPCPY
	lea	13(%edx), %eax
# else
	movl	%edx, %eax
# endif
	RETURN1

	.p2align 4
L(ExitTail15):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	movlpd	7(%ecx), %xmm0
	movlpd	%xmm0, 7(%edx)
# ifdef USE_AS_STPCPY
	lea	14(%edx), %eax
# else
	movl	%edx, %eax
# endif
	RETURN1

	.p2align 4
L(ExitTail16):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	movlpd	8(%ecx), %xmm0
	movlpd	%xmm0, 8(%edx)
# ifdef USE_AS_STPCPY
	lea	15(%edx), %eax
# else
	movl	%edx, %eax
# endif
	RETURN1

END (STRCPY)
# endif

#endif

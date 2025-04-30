/* strcpy with SSSE3
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

# ifndef USE_AS_STRCAT
#  include <sysdep.h>

#  define CFI_PUSH(REG)	\
	cfi_adjust_cfa_offset (4);	\
	cfi_rel_offset (REG, 0)

#  define CFI_POP(REG)	\
	cfi_adjust_cfa_offset (-4);	\
	cfi_restore (REG)

#  define PUSH(REG)	pushl REG; CFI_PUSH (REG)
#  define POP(REG)	popl REG; CFI_POP (REG)

#  ifndef STRCPY
#   define STRCPY  __strcpy_ssse3
#  endif

#  ifdef USE_AS_STRNCPY
#   define PARMS  8
#   define ENTRANCE PUSH (%ebx)
#   define RETURN  POP (%ebx); ret; CFI_PUSH (%ebx);
#   define RETURN1  POP (%edi); POP (%ebx); ret; CFI_PUSH (%ebx); CFI_PUSH (%edi)
#  else
#   define PARMS  4
#   define ENTRANCE
#   define RETURN  ret
#   define RETURN1  POP (%edi); ret; CFI_PUSH (%edi)
#  endif

#  ifdef USE_AS_STPCPY
#   define SAVE_RESULT(n)  lea	n(%edx), %eax
#   define SAVE_RESULT_TAIL(n)  lea	n(%edx), %eax
#  else
#   define SAVE_RESULT(n)  movl	%edi, %eax
#   define SAVE_RESULT_TAIL(n)  movl	%edx, %eax
#  endif

#  define STR1  PARMS
#  define STR2  STR1+4
#  define LEN  STR2+4

/* In this code following instructions are used for copying:
	movb	- 1 byte
	movw	- 2 byte
	movl	- 4 byte
	movlpd	- 8 byte
	movaps	- 16 byte - requires 16 byte alignment
	of	sourse and destination adresses.
*/

.text
ENTRY (STRCPY)
	ENTRANCE
	mov	STR1(%esp), %edx
	mov	STR2(%esp), %ecx
#  ifdef USE_AS_STRNCPY
	movl	LEN(%esp), %ebx
	cmp	$8, %ebx
	jbe	L(StrncpyExit8Bytes)
#  endif
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
#  ifdef USE_AS_STRNCPY
	cmp	$16, %ebx
	jb	L(StrncpyExit15Bytes)
#  endif
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
#  ifdef USE_AS_STRNCPY
	cmp	$16, %ebx
	je	L(ExitTail16)
#  endif
	cmpb	$0, 15(%ecx)
	jz	L(ExitTail16)

	PUSH	(%edi)
	mov	%edx, %edi
# endif
	PUSH	(%esi)
# ifdef USE_AS_STRNCPY
	mov	%ecx, %esi
	sub	$16, %ebx
	and	$0xf, %esi

/* add 16 bytes ecx_offset to ebx */

	add	%esi, %ebx
# endif
	lea	16(%ecx), %esi
	and	$-16, %esi
	pxor	%xmm0, %xmm0
	movlpd	(%ecx), %xmm1
	movlpd	%xmm1, (%edx)

	pcmpeqb	(%esi), %xmm0
	movlpd	8(%ecx), %xmm1
	movlpd	%xmm1, 8(%edx)

	pmovmskb %xmm0, %eax
	sub	%ecx, %esi

# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	mov	%edx, %eax
	lea	16(%edx), %edx
	and	$-16, %edx
	sub	%edx, %eax

# ifdef USE_AS_STRNCPY
	add	%eax, %esi
	lea	-1(%esi), %esi
	and	$1<<31, %esi
	test	%esi, %esi
	jnz	L(ContinueCopy)
	lea	16(%ebx), %ebx

L(ContinueCopy):
# endif
	sub	%eax, %ecx
	mov	%ecx, %eax
	and	$0xf, %eax
	mov	$0, %esi

/* case: ecx_offset == edx_offset */

	jz	L(Align16Both)

	cmp	$8, %eax
	jae	L(ShlHigh8)
	cmp	$1, %eax
	je	L(Shl1)
	cmp	$2, %eax
	je	L(Shl2)
	cmp	$3, %eax
	je	L(Shl3)
	cmp	$4, %eax
	je	L(Shl4)
	cmp	$5, %eax
	je	L(Shl5)
	cmp	$6, %eax
	je	L(Shl6)
	jmp	L(Shl7)

L(ShlHigh8):
	je	L(Shl8)
	cmp	$9, %eax
	je	L(Shl9)
	cmp	$10, %eax
	je	L(Shl10)
	cmp	$11, %eax
	je	L(Shl11)
	cmp	$12, %eax
	je	L(Shl12)
	cmp	$13, %eax
	je	L(Shl13)
	cmp	$14, %eax
	je	L(Shl14)
	jmp	L(Shl15)

L(Align16Both):
	movaps	(%ecx), %xmm1
	movaps	16(%ecx), %xmm2
	movaps	%xmm1, (%edx)
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	lea	16(%esi), %esi
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%ecx, %esi), %xmm3
	movaps	%xmm2, (%edx, %esi)
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %eax
	lea	16(%esi), %esi
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%ecx, %esi), %xmm4
	movaps	%xmm3, (%edx, %esi)
	pcmpeqb	%xmm4, %xmm0
	pmovmskb %xmm0, %eax
	lea	16(%esi), %esi
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%ecx, %esi), %xmm1
	movaps	%xmm4, (%edx, %esi)
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	lea	16(%esi), %esi
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%ecx, %esi), %xmm2
	movaps	%xmm1, (%edx, %esi)
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	lea	16(%esi), %esi
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%ecx, %esi), %xmm3
	movaps	%xmm2, (%edx, %esi)
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %eax
	lea	16(%esi), %esi
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	%xmm3, (%edx, %esi)
	mov	%ecx, %eax
	lea	16(%ecx, %esi), %ecx
	and	$-0x40, %ecx
	sub	%ecx, %eax
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	lea	112(%ebx, %eax), %ebx
# endif
	mov	$-0x40, %esi

L(Aligned64Loop):
	movaps	(%ecx), %xmm2
	movaps	32(%ecx), %xmm3
	movaps	%xmm2, %xmm4
	movaps	16(%ecx), %xmm5
	movaps	%xmm3, %xmm6
	movaps	48(%ecx), %xmm7
	pminub	%xmm5, %xmm2
	pminub	%xmm7, %xmm3
	pminub	%xmm2, %xmm3
	lea	64(%edx), %edx
	pcmpeqb	%xmm0, %xmm3
	lea	64(%ecx), %ecx
	pmovmskb %xmm3, %eax
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeaveCase2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Aligned64Leave)
	movaps	%xmm4, -64(%edx)
	movaps	%xmm5, -48(%edx)
	movaps	%xmm6, -32(%edx)
	movaps	%xmm7, -16(%edx)
	jmp	L(Aligned64Loop)

L(Aligned64Leave):
# ifdef USE_AS_STRNCPY
	lea	48(%ebx), %ebx
# endif
	pcmpeqb	%xmm4, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqb	%xmm5, %xmm0
# ifdef USE_AS_STRNCPY
	lea	-16(%ebx), %ebx
# endif
	pmovmskb %xmm0, %eax
	movaps	%xmm4, -64(%edx)
	test	%eax, %eax
	lea	16(%esi), %esi
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqb	%xmm6, %xmm0
# ifdef USE_AS_STRNCPY
	lea	-16(%ebx), %ebx
# endif
	pmovmskb %xmm0, %eax
	movaps	%xmm5, -48(%edx)
	test	%eax, %eax
	lea	16(%esi), %esi
	jnz	L(CopyFrom1To16Bytes)

	movaps	%xmm6, -32(%edx)
	pcmpeqb	%xmm7, %xmm0
# ifdef USE_AS_STRNCPY
	lea	-16(%ebx), %ebx
# endif
	pmovmskb %xmm0, %eax
	lea	16(%esi), %esi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl1):
	movaps	-1(%ecx), %xmm1
	movaps	15(%ecx), %xmm2
L(Shl1Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit1Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl1LoopExit)

	palignr	$1, %xmm1, %xmm2
	movaps	%xmm3, %xmm1
	movaps	%xmm2, (%edx)
	movaps	31(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit1Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl1LoopExit)

	palignr	$1, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	31(%ecx), %xmm2
	movaps	%xmm3, %xmm1

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit1Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl1LoopExit)

	palignr	$1, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	31(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit1Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl1LoopExit)

	palignr	$1, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	lea	31(%ecx), %ecx
	lea	16(%edx), %edx

	mov	%ecx, %eax
	and	$-0x40, %ecx
	sub	%ecx, %eax
	lea	-15(%ecx), %ecx
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	add	%eax, %ebx
# endif
	movaps	-1(%ecx), %xmm1

L(Shl1LoopStart):
	movaps	15(%ecx), %xmm2
	movaps	31(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	47(%ecx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	63(%ecx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$1, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$1, %xmm3, %xmm4
	jnz	L(Shl1Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeave1)
# endif
	palignr	$1, %xmm2, %xmm3
	lea	64(%ecx), %ecx
	palignr	$1, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%edx)
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	jmp	L(Shl1LoopStart)

L(Shl1LoopExit):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	movlpd	7(%ecx), %xmm0
	movlpd	%xmm0, 7(%edx)
	mov	$15, %esi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl2):
	movaps	-2(%ecx), %xmm1
	movaps	14(%ecx), %xmm2
L(Shl2Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit2Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl2LoopExit)

	palignr	$2, %xmm1, %xmm2
	movaps	%xmm3, %xmm1
	movaps	%xmm2, (%edx)
	movaps	30(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit2Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl2LoopExit)

	palignr	$2, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	30(%ecx), %xmm2
	movaps	%xmm3, %xmm1

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit2Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl2LoopExit)

	palignr	$2, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	30(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit2Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl2LoopExit)

	palignr	$2, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	lea	30(%ecx), %ecx
	lea	16(%edx), %edx

	mov	%ecx, %eax
	and	$-0x40, %ecx
	sub	%ecx, %eax
	lea	-14(%ecx), %ecx
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	add	%eax, %ebx
# endif
	movaps	-2(%ecx), %xmm1

L(Shl2LoopStart):
	movaps	14(%ecx), %xmm2
	movaps	30(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	46(%ecx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	62(%ecx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$2, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$2, %xmm3, %xmm4
	jnz	L(Shl2Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeave2)
# endif
	palignr	$2, %xmm2, %xmm3
	lea	64(%ecx), %ecx
	palignr	$2, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%edx)
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	jmp	L(Shl2LoopStart)

L(Shl2LoopExit):
	movlpd	(%ecx), %xmm0
	movlpd	6(%ecx), %xmm1
	movlpd	%xmm0, (%edx)
	movlpd	%xmm1, 6(%edx)
	mov	$14, %esi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl3):
	movaps	-3(%ecx), %xmm1
	movaps	13(%ecx), %xmm2
L(Shl3Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit3Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl3LoopExit)

	palignr	$3, %xmm1, %xmm2
	movaps	%xmm3, %xmm1
	movaps	%xmm2, (%edx)
	movaps	29(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit3Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl3LoopExit)

	palignr	$3, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	29(%ecx), %xmm2
	movaps	%xmm3, %xmm1

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit3Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl3LoopExit)

	palignr	$3, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	29(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit3Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl3LoopExit)

	palignr	$3, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	lea	29(%ecx), %ecx
	lea	16(%edx), %edx

	mov	%ecx, %eax
	and	$-0x40, %ecx
	sub	%ecx, %eax
	lea	-13(%ecx), %ecx
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	add	%eax, %ebx
# endif
	movaps	-3(%ecx), %xmm1

L(Shl3LoopStart):
	movaps	13(%ecx), %xmm2
	movaps	29(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	45(%ecx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	61(%ecx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$3, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$3, %xmm3, %xmm4
	jnz	L(Shl3Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeave3)
# endif
	palignr	$3, %xmm2, %xmm3
	lea	64(%ecx), %ecx
	palignr	$3, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%edx)
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	jmp	L(Shl3LoopStart)

L(Shl3LoopExit):
	movlpd	(%ecx), %xmm0
	movlpd	5(%ecx), %xmm1
	movlpd	%xmm0, (%edx)
	movlpd	%xmm1, 5(%edx)
	mov	$13, %esi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl4):
	movaps	-4(%ecx), %xmm1
	movaps	12(%ecx), %xmm2
L(Shl4Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit4Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl4LoopExit)

	palignr	$4, %xmm1, %xmm2
	movaps	%xmm3, %xmm1
	movaps	%xmm2, (%edx)
	movaps	28(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit4Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl4LoopExit)

	palignr	$4, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	28(%ecx), %xmm2
	movaps	%xmm3, %xmm1

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit4Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl4LoopExit)

	palignr	$4, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	28(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit4Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl4LoopExit)

	palignr	$4, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	lea	28(%ecx), %ecx
	lea	16(%edx), %edx

	mov	%ecx, %eax
	and	$-0x40, %ecx
	sub	%ecx, %eax
	lea	-12(%ecx), %ecx
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	add	%eax, %ebx
# endif
	movaps	-4(%ecx), %xmm1

L(Shl4LoopStart):
	movaps	12(%ecx), %xmm2
	movaps	28(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	44(%ecx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	60(%ecx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$4, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$4, %xmm3, %xmm4
	jnz	L(Shl4Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeave4)
# endif
	palignr	$4, %xmm2, %xmm3
	lea	64(%ecx), %ecx
	palignr	$4, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%edx)
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	jmp	L(Shl4LoopStart)

L(Shl4LoopExit):
	movlpd	(%ecx), %xmm0
	movl	8(%ecx), %esi
	movlpd	%xmm0, (%edx)
	movl	%esi, 8(%edx)
	mov	$12, %esi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl5):
	movaps	-5(%ecx), %xmm1
	movaps	11(%ecx), %xmm2
L(Shl5Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit5Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl5LoopExit)

	palignr	$5, %xmm1, %xmm2
	movaps	%xmm3, %xmm1
	movaps	%xmm2, (%edx)
	movaps	27(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit5Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl5LoopExit)

	palignr	$5, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	27(%ecx), %xmm2
	movaps	%xmm3, %xmm1

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit5Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl5LoopExit)

	palignr	$5, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	27(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit5Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl5LoopExit)

	palignr	$5, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	lea	27(%ecx), %ecx
	lea	16(%edx), %edx

	mov	%ecx, %eax
	and	$-0x40, %ecx
	sub	%ecx, %eax
	lea	-11(%ecx), %ecx
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	add	%eax, %ebx
# endif
	movaps	-5(%ecx), %xmm1

L(Shl5LoopStart):
	movaps	11(%ecx), %xmm2
	movaps	27(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	43(%ecx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	59(%ecx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$5, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$5, %xmm3, %xmm4
	jnz	L(Shl5Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeave5)
# endif
	palignr	$5, %xmm2, %xmm3
	lea	64(%ecx), %ecx
	palignr	$5, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%edx)
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	jmp	L(Shl5LoopStart)

L(Shl5LoopExit):
	movlpd	(%ecx), %xmm0
	movl	7(%ecx), %esi
	movlpd	%xmm0, (%edx)
	movl	%esi, 7(%edx)
	mov	$11, %esi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl6):
	movaps	-6(%ecx), %xmm1
	movaps	10(%ecx), %xmm2
L(Shl6Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit6Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl6LoopExit)

	palignr	$6, %xmm1, %xmm2
	movaps	%xmm3, %xmm1
	movaps	%xmm2, (%edx)
	movaps	26(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit6Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl6LoopExit)

	palignr	$6, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	26(%ecx), %xmm2
	movaps	%xmm3, %xmm1

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit6Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl6LoopExit)

	palignr	$6, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	26(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit6Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl6LoopExit)

	palignr	$6, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	lea	26(%ecx), %ecx
	lea	16(%edx), %edx

	mov	%ecx, %eax
	and	$-0x40, %ecx
	sub	%ecx, %eax
	lea	-10(%ecx), %ecx
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	add	%eax, %ebx
# endif
	movaps	-6(%ecx), %xmm1

L(Shl6LoopStart):
	movaps	10(%ecx), %xmm2
	movaps	26(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	42(%ecx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	58(%ecx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$6, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$6, %xmm3, %xmm4
	jnz	L(Shl6Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeave6)
# endif
	palignr	$6, %xmm2, %xmm3
	lea	64(%ecx), %ecx
	palignr	$6, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%edx)
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	jmp	L(Shl6LoopStart)

L(Shl6LoopExit):
	movlpd	(%ecx), %xmm0
	movl	6(%ecx), %esi
	movlpd	%xmm0, (%edx)
	movl	%esi, 6(%edx)
	mov	$10, %esi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl7):
	movaps	-7(%ecx), %xmm1
	movaps	9(%ecx), %xmm2
L(Shl7Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit7Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl7LoopExit)

	palignr	$7, %xmm1, %xmm2
	movaps	%xmm3, %xmm1
	movaps	%xmm2, (%edx)
	movaps	25(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit7Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl7LoopExit)

	palignr	$7, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	25(%ecx), %xmm2
	movaps	%xmm3, %xmm1

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit7Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl7LoopExit)

	palignr	$7, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	25(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit7Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl7LoopExit)

	palignr	$7, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	lea	25(%ecx), %ecx
	lea	16(%edx), %edx

	mov	%ecx, %eax
	and	$-0x40, %ecx
	sub	%ecx, %eax
	lea	-9(%ecx), %ecx
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	add	%eax, %ebx
# endif
	movaps	-7(%ecx), %xmm1

L(Shl7LoopStart):
	movaps	9(%ecx), %xmm2
	movaps	25(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	41(%ecx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	57(%ecx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$7, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$7, %xmm3, %xmm4
	jnz	L(Shl7Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeave7)
# endif
	palignr	$7, %xmm2, %xmm3
	lea	64(%ecx), %ecx
	palignr	$7, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%edx)
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	jmp	L(Shl7LoopStart)

L(Shl7LoopExit):
	movlpd	(%ecx), %xmm0
	movl	5(%ecx), %esi
	movlpd	%xmm0, (%edx)
	movl	%esi, 5(%edx)
	mov	$9, %esi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl8):
	movaps	-8(%ecx), %xmm1
	movaps	8(%ecx), %xmm2
L(Shl8Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit8Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl8LoopExit)

	palignr	$8, %xmm1, %xmm2
	movaps	%xmm3, %xmm1
	movaps	%xmm2, (%edx)
	movaps	24(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit8Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl8LoopExit)

	palignr	$8, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	24(%ecx), %xmm2
	movaps	%xmm3, %xmm1

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit8Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl8LoopExit)

	palignr	$8, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	24(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit8Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl8LoopExit)

	palignr	$8, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	lea	24(%ecx), %ecx
	lea	16(%edx), %edx

	mov	%ecx, %eax
	and	$-0x40, %ecx
	sub	%ecx, %eax
	lea	-8(%ecx), %ecx
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	add	%eax, %ebx
# endif
	movaps	-8(%ecx), %xmm1

L(Shl8LoopStart):
	movaps	8(%ecx), %xmm2
	movaps	24(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	40(%ecx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	56(%ecx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$8, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$8, %xmm3, %xmm4
	jnz	L(Shl8Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeave8)
# endif
	palignr	$8, %xmm2, %xmm3
	lea	64(%ecx), %ecx
	palignr	$8, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%edx)
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	jmp	L(Shl8LoopStart)

L(Shl8LoopExit):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	mov	$8, %esi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl9):
	movaps	-9(%ecx), %xmm1
	movaps	7(%ecx), %xmm2
L(Shl9Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit9Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl9LoopExit)

	palignr	$9, %xmm1, %xmm2
	movaps	%xmm3, %xmm1
	movaps	%xmm2, (%edx)
	movaps	23(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit9Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl9LoopExit)

	palignr	$9, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	23(%ecx), %xmm2
	movaps	%xmm3, %xmm1

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit9Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl9LoopExit)

	palignr	$9, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	23(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit9Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl9LoopExit)

	palignr	$9, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	lea	23(%ecx), %ecx
	lea	16(%edx), %edx

	mov	%ecx, %eax
	and	$-0x40, %ecx
	sub	%ecx, %eax
	lea	-7(%ecx), %ecx
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	add	%eax, %ebx
# endif
	movaps	-9(%ecx), %xmm1

L(Shl9LoopStart):
	movaps	7(%ecx), %xmm2
	movaps	23(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	39(%ecx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	55(%ecx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$9, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$9, %xmm3, %xmm4
	jnz	L(Shl9Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeave9)
# endif
	palignr	$9, %xmm2, %xmm3
	lea	64(%ecx), %ecx
	palignr	$9, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%edx)
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	jmp	L(Shl9LoopStart)

L(Shl9LoopExit):
	movlpd	-1(%ecx), %xmm0
	movlpd	%xmm0, -1(%edx)
	mov	$7, %esi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl10):
	movaps	-10(%ecx), %xmm1
	movaps	6(%ecx), %xmm2
L(Shl10Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit10Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl10LoopExit)

	palignr	$10, %xmm1, %xmm2
	movaps	%xmm3, %xmm1
	movaps	%xmm2, (%edx)
	movaps	22(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit10Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl10LoopExit)

	palignr	$10, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	22(%ecx), %xmm2
	movaps	%xmm3, %xmm1

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit10Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl10LoopExit)

	palignr	$10, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	22(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit10Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl10LoopExit)

	palignr	$10, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	lea	22(%ecx), %ecx
	lea	16(%edx), %edx

	mov	%ecx, %eax
	and	$-0x40, %ecx
	sub	%ecx, %eax
	lea	-6(%ecx), %ecx
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	add	%eax, %ebx
# endif
	movaps	-10(%ecx), %xmm1

L(Shl10LoopStart):
	movaps	6(%ecx), %xmm2
	movaps	22(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	38(%ecx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	54(%ecx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$10, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$10, %xmm3, %xmm4
	jnz	L(Shl10Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeave10)
# endif
	palignr	$10, %xmm2, %xmm3
	lea	64(%ecx), %ecx
	palignr	$10, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%edx)
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	jmp	L(Shl10LoopStart)

L(Shl10LoopExit):
	movlpd	-2(%ecx), %xmm0
	movlpd	%xmm0, -2(%edx)
	mov	$6, %esi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl11):
	movaps	-11(%ecx), %xmm1
	movaps	5(%ecx), %xmm2
L(Shl11Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit11Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl11LoopExit)

	palignr	$11, %xmm1, %xmm2
	movaps	%xmm3, %xmm1
	movaps	%xmm2, (%edx)
	movaps	21(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit11Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl11LoopExit)

	palignr	$11, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	21(%ecx), %xmm2
	movaps	%xmm3, %xmm1

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit11Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl11LoopExit)

	palignr	$11, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	21(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit11Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl11LoopExit)

	palignr	$11, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	lea	21(%ecx), %ecx
	lea	16(%edx), %edx

	mov	%ecx, %eax
	and	$-0x40, %ecx
	sub	%ecx, %eax
	lea	-5(%ecx), %ecx
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	add	%eax, %ebx
# endif
	movaps	-11(%ecx), %xmm1

L(Shl11LoopStart):
	movaps	5(%ecx), %xmm2
	movaps	21(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	37(%ecx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	53(%ecx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$11, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$11, %xmm3, %xmm4
	jnz	L(Shl11Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeave11)
# endif
	palignr	$11, %xmm2, %xmm3
	lea	64(%ecx), %ecx
	palignr	$11, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%edx)
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	jmp	L(Shl11LoopStart)

L(Shl11LoopExit):
	movlpd	-3(%ecx), %xmm0
	movlpd	%xmm0, -3(%edx)
	mov	$5, %esi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl12):
	movaps	-12(%ecx), %xmm1
	movaps	4(%ecx), %xmm2
L(Shl12Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit12Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl12LoopExit)

	palignr	$12, %xmm1, %xmm2
	movaps	%xmm3, %xmm1
	movaps	%xmm2, (%edx)
	movaps	20(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit12Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl12LoopExit)

	palignr	$12, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	20(%ecx), %xmm2
	movaps	%xmm3, %xmm1

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit12Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl12LoopExit)

	palignr	$12, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	20(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit12Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl12LoopExit)

	palignr	$12, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	lea	20(%ecx), %ecx
	lea	16(%edx), %edx

	mov	%ecx, %eax
	and	$-0x40, %ecx
	sub	%ecx, %eax
	lea	-4(%ecx), %ecx
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	add	%eax, %ebx
# endif
	movaps	-12(%ecx), %xmm1

L(Shl12LoopStart):
	movaps	4(%ecx), %xmm2
	movaps	20(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	36(%ecx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	52(%ecx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$12, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$12, %xmm3, %xmm4
	jnz	L(Shl12Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeave12)
# endif
	palignr	$12, %xmm2, %xmm3
	lea	64(%ecx), %ecx
	palignr	$12, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%edx)
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	jmp	L(Shl12LoopStart)

L(Shl12LoopExit):
	movl	(%ecx), %esi
	movl	%esi, (%edx)
	mov	$4, %esi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl13):
	movaps	-13(%ecx), %xmm1
	movaps	3(%ecx), %xmm2
L(Shl13Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit13Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl13LoopExit)

	palignr	$13, %xmm1, %xmm2
	movaps	%xmm3, %xmm1
	movaps	%xmm2, (%edx)
	movaps	19(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit13Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl13LoopExit)

	palignr	$13, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	19(%ecx), %xmm2
	movaps	%xmm3, %xmm1

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit13Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl13LoopExit)

	palignr	$13, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	19(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit13Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl13LoopExit)

	palignr	$13, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	lea	19(%ecx), %ecx
	lea	16(%edx), %edx

	mov	%ecx, %eax
	and	$-0x40, %ecx
	sub	%ecx, %eax
	lea	-3(%ecx), %ecx
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	add	%eax, %ebx
# endif
	movaps	-13(%ecx), %xmm1

L(Shl13LoopStart):
	movaps	3(%ecx), %xmm2
	movaps	19(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	35(%ecx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	51(%ecx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$13, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$13, %xmm3, %xmm4
	jnz	L(Shl13Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeave13)
# endif
	palignr	$13, %xmm2, %xmm3
	lea	64(%ecx), %ecx
	palignr	$13, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%edx)
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	jmp	L(Shl13LoopStart)

L(Shl13LoopExit):
	movl	-1(%ecx), %esi
	movl	%esi, -1(%edx)
	mov	$3, %esi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl14):
	movaps	-14(%ecx), %xmm1
	movaps	2(%ecx), %xmm2
L(Shl14Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit14Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl14LoopExit)

	palignr	$14, %xmm1, %xmm2
	movaps	%xmm3, %xmm1
	movaps	%xmm2, (%edx)
	movaps	18(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit14Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl14LoopExit)

	palignr	$14, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	18(%ecx), %xmm2
	movaps	%xmm3, %xmm1

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit14Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl14LoopExit)

	palignr	$14, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	18(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit14Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl14LoopExit)

	palignr	$14, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	lea	18(%ecx), %ecx
	lea	16(%edx), %edx

	mov	%ecx, %eax
	and	$-0x40, %ecx
	sub	%ecx, %eax
	lea	-2(%ecx), %ecx
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	add	%eax, %ebx
# endif
	movaps	-14(%ecx), %xmm1

L(Shl14LoopStart):
	movaps	2(%ecx), %xmm2
	movaps	18(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	34(%ecx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	50(%ecx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$14, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$14, %xmm3, %xmm4
	jnz	L(Shl14Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeave14)
# endif
	palignr	$14, %xmm2, %xmm3
	lea	64(%ecx), %ecx
	palignr	$14, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%edx)
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	jmp	L(Shl14LoopStart)

L(Shl14LoopExit):
	movl	-2(%ecx), %esi
	movl	%esi, -2(%edx)
	mov	$2, %esi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl15):
	movaps	-15(%ecx), %xmm1
	movaps	1(%ecx), %xmm2
L(Shl15Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit15Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl15LoopExit)

	palignr	$15, %xmm1, %xmm2
	movaps	%xmm3, %xmm1
	movaps	%xmm2, (%edx)
	movaps	17(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit15Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl15LoopExit)

	palignr	$15, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	17(%ecx), %xmm2
	movaps	%xmm3, %xmm1

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit15Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl15LoopExit)

	palignr	$15, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	17(%ecx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
# ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	jbe	L(StrncpyExit15Case2OrCase3)
# endif
	test	%eax, %eax
	jnz	L(Shl15LoopExit)

	palignr	$15, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	lea	17(%ecx), %ecx
	lea	16(%edx), %edx

	mov	%ecx, %eax
	and	$-0x40, %ecx
	sub	%ecx, %eax
	lea	-1(%ecx), %ecx
	sub	%eax, %edx
# ifdef USE_AS_STRNCPY
	add	%eax, %ebx
# endif
	movaps	-15(%ecx), %xmm1

L(Shl15LoopStart):
	movaps	1(%ecx), %xmm2
	movaps	17(%ecx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	33(%ecx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	49(%ecx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$15, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$15, %xmm3, %xmm4
	jnz	L(Shl15Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %ebx
	jbe	L(StrncpyLeave15)
# endif
	palignr	$15, %xmm2, %xmm3
	lea	64(%ecx), %ecx
	palignr	$15, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%edx)
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	jmp	L(Shl15LoopStart)

L(Shl15LoopExit):
	movl	-3(%ecx), %esi
	movl	%esi, -3(%edx)
	mov	$1, %esi
# ifdef USE_AS_STRCAT
	jmp	L(CopyFrom1To16Bytes)
# endif


# ifndef USE_AS_STRCAT

	.p2align 4
L(CopyFrom1To16Bytes):
#  ifdef USE_AS_STRNCPY
	add	$16, %ebx
#  endif
	add	%esi, %edx
	add	%esi, %ecx

	POP	(%esi)
	test	%al, %al
	jz	L(ExitHigh8)

L(CopyFrom1To16BytesLess8):
	mov	%al, %ah
	and	$15, %ah
	jz	L(ExitHigh4)

	test	$0x01, %al
	jnz	L(Exit1)
	test	$0x02, %al
	jnz	L(Exit2)
	test	$0x04, %al
	jnz	L(Exit3)

	.p2align 4
L(Exit4):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	SAVE_RESULT	(3)
#  ifdef USE_AS_STRNCPY
	sub	$4, %ebx
	lea	4(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

	.p2align 4
L(ExitHigh4):
	test	$0x10, %al
	jnz	L(Exit5)
	test	$0x20, %al
	jnz	L(Exit6)
	test	$0x40, %al
	jnz	L(Exit7)

	.p2align 4
L(Exit8):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	SAVE_RESULT	(7)
#  ifdef USE_AS_STRNCPY
	sub	$8, %ebx
	lea	8(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

	.p2align 4
L(ExitHigh8):
	mov	%ah, %al
	and	$15, %al
	jz	L(ExitHigh12)

	test	$0x01, %ah
	jnz	L(Exit9)
	test	$0x02, %ah
	jnz	L(Exit10)
	test	$0x04, %ah
	jnz	L(Exit11)

	.p2align 4
L(Exit12):
	movlpd	(%ecx), %xmm0
	movl	8(%ecx), %eax
	movlpd	%xmm0, (%edx)
	movl	%eax, 8(%edx)
	SAVE_RESULT	(11)
#  ifdef USE_AS_STRNCPY
	sub	$12, %ebx
	lea	12(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

	.p2align 4
L(ExitHigh12):
	test	$0x10, %ah
	jnz	L(Exit13)
	test	$0x20, %ah
	jnz	L(Exit14)
	test	$0x40, %ah
	jnz	L(Exit15)

	.p2align 4
L(Exit16):
	movdqu	(%ecx), %xmm0
	movdqu	%xmm0, (%edx)
	SAVE_RESULT	(15)
#  ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	lea	16(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

#   ifdef USE_AS_STRNCPY

	CFI_PUSH(%esi)

	.p2align 4
L(CopyFrom1To16BytesCase2):
	add	$16, %ebx
	add	%esi, %ecx
	add	%esi, %edx

	POP	(%esi)

	test	%al, %al
	jz	L(ExitHighCase2)

	cmp	$8, %ebx
	ja	L(CopyFrom1To16BytesLess8)

	test	$0x01, %al
	jnz	L(Exit1)
	cmp	$1, %ebx
	je	L(Exit1)
	test	$0x02, %al
	jnz	L(Exit2)
	cmp	$2, %ebx
	je	L(Exit2)
	test	$0x04, %al
	jnz	L(Exit3)
	cmp	$3, %ebx
	je	L(Exit3)
	test	$0x08, %al
	jnz	L(Exit4)
	cmp	$4, %ebx
	je	L(Exit4)
	test	$0x10, %al
	jnz	L(Exit5)
	cmp	$5, %ebx
	je	L(Exit5)
	test	$0x20, %al
	jnz	L(Exit6)
	cmp	$6, %ebx
	je	L(Exit6)
	test	$0x40, %al
	jnz	L(Exit7)
	cmp	$7, %ebx
	je	L(Exit7)
	jmp	L(Exit8)

	.p2align 4
L(ExitHighCase2):
	cmp	$8, %ebx
	jbe	L(CopyFrom1To16BytesLess8Case3)

	test	$0x01, %ah
	jnz	L(Exit9)
	cmp	$9, %ebx
	je	L(Exit9)
	test	$0x02, %ah
	jnz	L(Exit10)
	cmp	$10, %ebx
	je	L(Exit10)
	test	$0x04, %ah
	jnz	L(Exit11)
	cmp	$11, %ebx
	je	L(Exit11)
	test	$0x8, %ah
	jnz	L(Exit12)
	cmp	$12, %ebx
	je	L(Exit12)
	test	$0x10, %ah
	jnz	L(Exit13)
	cmp	$13, %ebx
	je	L(Exit13)
	test	$0x20, %ah
	jnz	L(Exit14)
	cmp	$14, %ebx
	je	L(Exit14)
	test	$0x40, %ah
	jnz	L(Exit15)
	cmp	$15, %ebx
	je	L(Exit15)
	jmp	L(Exit16)

	CFI_PUSH(%esi)

	.p2align 4
L(CopyFrom1To16BytesCase2OrCase3):
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)

	.p2align 4
L(CopyFrom1To16BytesCase3):
	add	$16, %ebx
	add	%esi, %edx
	add	%esi, %ecx

	POP	(%esi)

	cmp	$8, %ebx
	ja	L(ExitHigh8Case3)

L(CopyFrom1To16BytesLess8Case3):
	cmp	$4, %ebx
	ja	L(ExitHigh4Case3)

	cmp	$1, %ebx
	je	L(Exit1)
	cmp	$2, %ebx
	je	L(Exit2)
	cmp	$3, %ebx
	je	L(Exit3)
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	SAVE_RESULT	(4)
	RETURN1

	.p2align 4
L(ExitHigh4Case3):
	cmp	$5, %ebx
	je	L(Exit5)
	cmp	$6, %ebx
	je	L(Exit6)
	cmp	$7, %ebx
	je	L(Exit7)
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	SAVE_RESULT	(8)
	RETURN1

	.p2align 4
L(ExitHigh8Case3):
	cmp	$12, %ebx
	ja	L(ExitHigh12Case3)

	cmp	$9, %ebx
	je	L(Exit9)
	cmp	$10, %ebx
	je	L(Exit10)
	cmp	$11, %ebx
	je	L(Exit11)
	movlpd	(%ecx), %xmm0
	movl	8(%ecx), %eax
	movlpd	%xmm0, (%edx)
	movl	%eax, 8(%edx)
	SAVE_RESULT	(12)
	RETURN1

	.p2align 4
L(ExitHigh12Case3):
	cmp	$13, %ebx
	je	L(Exit13)
	cmp	$14, %ebx
	je	L(Exit14)
	cmp	$15, %ebx
	je	L(Exit15)
	movlpd	(%ecx), %xmm0
	movlpd	8(%ecx), %xmm1
	movlpd	%xmm0, (%edx)
	movlpd	%xmm1, 8(%edx)
	SAVE_RESULT	(16)
	RETURN1

#  endif

	.p2align 4
L(Exit1):
	movb	(%ecx), %al
	movb	%al, (%edx)
	SAVE_RESULT	(0)
#  ifdef USE_AS_STRNCPY
	sub	$1, %ebx
	lea	1(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

	.p2align 4
L(Exit2):
	movw	(%ecx), %ax
	movw	%ax, (%edx)
	SAVE_RESULT	(1)
#  ifdef USE_AS_STRNCPY
	sub	$2, %ebx
	lea	2(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

	.p2align 4
L(Exit3):
	movw	(%ecx), %ax
	movw	%ax, (%edx)
	movb	2(%ecx), %al
	movb	%al, 2(%edx)
	SAVE_RESULT	(2)
#  ifdef USE_AS_STRNCPY
	sub	$3, %ebx
	lea	3(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

	.p2align 4
L(Exit5):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movb	4(%ecx), %al
	movb	%al, 4(%edx)
	SAVE_RESULT	(4)
#  ifdef USE_AS_STRNCPY
	sub	$5, %ebx
	lea	5(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

	.p2align 4
L(Exit6):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movw	4(%ecx), %ax
	movw	%ax, 4(%edx)
	SAVE_RESULT	(5)
#  ifdef USE_AS_STRNCPY
	sub	$6, %ebx
	lea	6(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

	.p2align 4
L(Exit7):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	3(%ecx), %eax
	movl	%eax, 3(%edx)
	SAVE_RESULT	(6)
#  ifdef USE_AS_STRNCPY
	sub	$7, %ebx
	lea	7(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

	.p2align 4
L(Exit9):
	movlpd	(%ecx), %xmm0
	movb	8(%ecx), %al
	movlpd	%xmm0, (%edx)
	movb	%al, 8(%edx)
	SAVE_RESULT	(8)
#  ifdef USE_AS_STRNCPY
	sub	$9, %ebx
	lea	9(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

	.p2align 4
L(Exit10):
	movlpd	(%ecx), %xmm0
	movw	8(%ecx), %ax
	movlpd	%xmm0, (%edx)
	movw	%ax, 8(%edx)
	SAVE_RESULT	(9)
#  ifdef USE_AS_STRNCPY
	sub	$10, %ebx
	lea	10(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

	.p2align 4
L(Exit11):
	movlpd	(%ecx), %xmm0
	movl	7(%ecx), %eax
	movlpd	%xmm0, (%edx)
	movl	%eax, 7(%edx)
	SAVE_RESULT	(10)
#  ifdef USE_AS_STRNCPY
	sub	$11, %ebx
	lea	11(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

	.p2align 4
L(Exit13):
	movlpd	(%ecx), %xmm0
	movlpd	5(%ecx), %xmm1
	movlpd	%xmm0, (%edx)
	movlpd	%xmm1, 5(%edx)
	SAVE_RESULT	(12)
#  ifdef USE_AS_STRNCPY
	sub	$13, %ebx
	lea	13(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

	.p2align 4
L(Exit14):
	movlpd	(%ecx), %xmm0
	movlpd	6(%ecx), %xmm1
	movlpd	%xmm0, (%edx)
	movlpd	%xmm1, 6(%edx)
	SAVE_RESULT	(13)
#  ifdef USE_AS_STRNCPY
	sub	$14, %ebx
	lea	14(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

	.p2align 4
L(Exit15):
	movlpd	(%ecx), %xmm0
	movlpd	7(%ecx), %xmm1
	movlpd	%xmm0, (%edx)
	movlpd	%xmm1, 7(%edx)
	SAVE_RESULT	(14)
#  ifdef USE_AS_STRNCPY
	sub	$15, %ebx
	lea	15(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN1

CFI_POP	(%edi)

#  ifdef USE_AS_STRNCPY
	.p2align 4
L(Fill0):
	RETURN

	.p2align 4
L(Fill1):
	movb	%dl, (%ecx)
	RETURN

	.p2align 4
L(Fill2):
	movw	%dx, (%ecx)
	RETURN

	.p2align 4
L(Fill3):
	movw	%dx, (%ecx)
	movb	%dl, 2(%ecx)
	RETURN

	.p2align 4
L(Fill4):
	movl	%edx, (%ecx)
	RETURN

	.p2align 4
L(Fill5):
	movl	%edx, (%ecx)
	movb	%dl, 4(%ecx)
	RETURN

	.p2align 4
L(Fill6):
	movl	%edx, (%ecx)
	movw	%dx, 4(%ecx)
	RETURN

	.p2align 4
L(Fill7):
	movl	%edx, (%ecx)
	movl	%edx, 3(%ecx)
	RETURN

	.p2align 4
L(Fill8):
	movlpd	%xmm0, (%ecx)
	RETURN

	.p2align 4
L(Fill9):
	movlpd	%xmm0, (%ecx)
	movb	%dl, 8(%ecx)
	RETURN

	.p2align 4
L(Fill10):
	movlpd	%xmm0, (%ecx)
	movw	%dx, 8(%ecx)
	RETURN

	.p2align 4
L(Fill11):
	movlpd	%xmm0, (%ecx)
	movl	%edx, 7(%ecx)
	RETURN

	.p2align 4
L(Fill12):
	movlpd	%xmm0, (%ecx)
	movl	%edx, 8(%ecx)
	RETURN

	.p2align 4
L(Fill13):
	movlpd	%xmm0, (%ecx)
	movlpd	%xmm0, 5(%ecx)
	RETURN

	.p2align 4
L(Fill14):
	movlpd	%xmm0, (%ecx)
	movlpd	%xmm0, 6(%ecx)
	RETURN

	.p2align 4
L(Fill15):
	movlpd	%xmm0, (%ecx)
	movlpd	%xmm0, 7(%ecx)
	RETURN

	.p2align 4
L(Fill16):
	movlpd	%xmm0, (%ecx)
	movlpd	%xmm0, 8(%ecx)
	RETURN

	.p2align 4
L(StrncpyFillExit1):
	lea	16(%ebx), %ebx
L(FillFrom1To16Bytes):
	test	%ebx, %ebx
	jz	L(Fill0)
	cmp	$16, %ebx
	je	L(Fill16)
	cmp	$8, %ebx
	je	L(Fill8)
	jg	L(FillMore8)
	cmp	$4, %ebx
	je	L(Fill4)
	jg	L(FillMore4)
	cmp	$2, %ebx
	jl	L(Fill1)
	je	L(Fill2)
	jg	L(Fill3)
L(FillMore8):	/* but less than 16 */
	cmp	$12, %ebx
	je	L(Fill12)
	jl	L(FillLess12)
	cmp	$14, %ebx
	jl	L(Fill13)
	je	L(Fill14)
	jg	L(Fill15)
L(FillMore4):	/* but less than 8 */
	cmp	$6, %ebx
	jl	L(Fill5)
	je	L(Fill6)
	jg	L(Fill7)
L(FillLess12):	/* but more than 8 */
	cmp	$10, %ebx
	jl	L(Fill9)
	je	L(Fill10)
	jmp	L(Fill11)

	CFI_PUSH(%edi)

	.p2align 4
L(StrncpyFillTailWithZero1):
	POP	(%edi)
L(StrncpyFillTailWithZero):
	pxor	%xmm0, %xmm0
	xor	%edx, %edx
	sub	$16, %ebx
	jbe	L(StrncpyFillExit1)

	movlpd	%xmm0, (%ecx)
	movlpd	%xmm0, 8(%ecx)

	lea	16(%ecx), %ecx

	mov	%ecx, %edx
	and	$0xf, %edx
	sub	%edx, %ecx
	add	%edx, %ebx
	xor	%edx, %edx
	sub	$64, %ebx
	jb	L(StrncpyFillLess64)

L(StrncpyFillLoopMovdqa):
	movdqa	%xmm0, (%ecx)
	movdqa	%xmm0, 16(%ecx)
	movdqa	%xmm0, 32(%ecx)
	movdqa	%xmm0, 48(%ecx)
	lea	64(%ecx), %ecx
	sub	$64, %ebx
	jae	L(StrncpyFillLoopMovdqa)

L(StrncpyFillLess64):
	add	$32, %ebx
	jl	L(StrncpyFillLess32)
	movdqa	%xmm0, (%ecx)
	movdqa	%xmm0, 16(%ecx)
	lea	32(%ecx), %ecx
	sub	$16, %ebx
	jl	L(StrncpyFillExit1)
	movdqa	%xmm0, (%ecx)
	lea	16(%ecx), %ecx
	jmp	L(FillFrom1To16Bytes)

L(StrncpyFillLess32):
	add	$16, %ebx
	jl	L(StrncpyFillExit1)
	movdqa	%xmm0, (%ecx)
	lea	16(%ecx), %ecx
	jmp	L(FillFrom1To16Bytes)
#  endif

	.p2align 4
L(ExitTail1):
	movb	(%ecx), %al
	movb	%al, (%edx)
	SAVE_RESULT_TAIL (0)
#  ifdef USE_AS_STRNCPY
	sub	$1, %ebx
	lea	1(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN

	.p2align 4
L(ExitTail2):
	movw	(%ecx), %ax
	movw	%ax, (%edx)
	SAVE_RESULT_TAIL (1)
#  ifdef USE_AS_STRNCPY
	sub	$2, %ebx
	lea	2(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN

	.p2align 4
L(ExitTail3):
	movw	(%ecx), %ax
	movw	%ax, (%edx)
	movb	2(%ecx), %al
	movb	%al, 2(%edx)
	SAVE_RESULT_TAIL (2)
#  ifdef USE_AS_STRNCPY
	sub	$3, %ebx
	lea	3(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN

	.p2align 4
L(ExitTail4):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	SAVE_RESULT_TAIL (3)
#  ifdef USE_AS_STRNCPY
	sub	$4, %ebx
	lea	4(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN

	.p2align 4
L(ExitTail5):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movb	4(%ecx), %al
	movb	%al, 4(%edx)
	SAVE_RESULT_TAIL (4)
#  ifdef USE_AS_STRNCPY
	sub	$5, %ebx
	lea	5(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN

	.p2align 4
L(ExitTail6):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movw	4(%ecx), %ax
	movw	%ax, 4(%edx)
	SAVE_RESULT_TAIL (5)
#  ifdef USE_AS_STRNCPY
	sub	$6, %ebx
	lea	6(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN

	.p2align 4
L(ExitTail7):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	3(%ecx), %eax
	movl	%eax, 3(%edx)
	SAVE_RESULT_TAIL (6)
#  ifdef USE_AS_STRNCPY
	sub	$7, %ebx
	lea	7(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN

	.p2align 4
L(ExitTail8):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	SAVE_RESULT_TAIL (7)
#  ifdef USE_AS_STRNCPY
	sub	$8, %ebx
	lea	8(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#  endif
	RETURN

	.p2align 4
L(ExitTail9):
	movlpd	(%ecx), %xmm0
	movb	8(%ecx), %al
	movlpd	%xmm0, (%edx)
	movb	%al, 8(%edx)
	SAVE_RESULT_TAIL (8)
#  ifdef USE_AS_STRNCPY
	sub	$9, %ebx
	lea	9(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN

	.p2align 4
L(ExitTail10):
	movlpd	(%ecx), %xmm0
	movw	8(%ecx), %ax
	movlpd	%xmm0, (%edx)
	movw	%ax, 8(%edx)
	SAVE_RESULT_TAIL (9)
#  ifdef USE_AS_STRNCPY
	sub	$10, %ebx
	lea	10(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN

	.p2align 4
L(ExitTail11):
	movlpd	(%ecx), %xmm0
	movl	7(%ecx), %eax
	movlpd	%xmm0, (%edx)
	movl	%eax, 7(%edx)
	SAVE_RESULT_TAIL (10)
#  ifdef USE_AS_STRNCPY
	sub	$11, %ebx
	lea	11(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN

	.p2align 4
L(ExitTail12):
	movlpd	(%ecx), %xmm0
	movl	8(%ecx), %eax
	movlpd	%xmm0, (%edx)
	movl	%eax, 8(%edx)
	SAVE_RESULT_TAIL (11)
#  ifdef USE_AS_STRNCPY
	sub	$12, %ebx
	lea	12(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN

	.p2align 4
L(ExitTail13):
	movlpd	(%ecx), %xmm0
	movlpd	5(%ecx), %xmm1
	movlpd	%xmm0, (%edx)
	movlpd	%xmm1, 5(%edx)
	SAVE_RESULT_TAIL (12)
#  ifdef USE_AS_STRNCPY
	sub	$13, %ebx
	lea	13(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN

	.p2align 4
L(ExitTail14):
	movlpd	(%ecx), %xmm0
	movlpd	6(%ecx), %xmm1
	movlpd	%xmm0, (%edx)
	movlpd	%xmm1, 6(%edx)
	SAVE_RESULT_TAIL (13)
#  ifdef USE_AS_STRNCPY
	sub	$14, %ebx
	lea	14(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#  ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN

	.p2align 4
L(ExitTail15):
	movlpd	(%ecx), %xmm0
	movlpd	7(%ecx), %xmm1
	movlpd	%xmm0, (%edx)
	movlpd	%xmm1, 7(%edx)
	SAVE_RESULT_TAIL (14)
#  ifdef USE_AS_STRNCPY
	sub	$15, %ebx
	lea	15(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#  endif
	RETURN

	.p2align 4
L(ExitTail16):
	movdqu	(%ecx), %xmm0
	movdqu	%xmm0, (%edx)
	SAVE_RESULT_TAIL (15)
#  ifdef USE_AS_STRNCPY
	sub	$16, %ebx
	lea	16(%edx), %ecx
	jnz	L(StrncpyFillTailWithZero)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
#  endif
	RETURN
# endif

# ifdef USE_AS_STRNCPY
#  ifndef USE_AS_STRCAT
	CFI_PUSH (%esi)
	CFI_PUSH (%edi)
#  endif
	.p2align 4
L(StrncpyLeaveCase2OrCase3):
	test	%eax, %eax
	jnz	L(Aligned64LeaveCase2)

L(Aligned64LeaveCase3):
	add	$48, %ebx
	jle	L(CopyFrom1To16BytesCase3)
	movaps	%xmm4, -64(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase3)
	movaps	%xmm5, -48(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase3)
	movaps	%xmm6, -32(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
	jmp	L(CopyFrom1To16BytesCase3)

L(Aligned64LeaveCase2):
	pcmpeqb	%xmm4, %xmm0
	pmovmskb %xmm0, %eax
	add	$48, %ebx
	jle	L(CopyFrom1To16BytesCase2OrCase3)
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqb	%xmm5, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm4, -64(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqb	%xmm6, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm5, -48(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqb	%xmm7, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm6, -32(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
	jmp	L(CopyFrom1To16BytesCase2)

/*--------------------------------------------------*/
	.p2align 4
L(StrncpyExit1Case2OrCase3):
	movlpd	(%ecx), %xmm0
	movlpd	7(%ecx), %xmm1
	movlpd	%xmm0, (%edx)
	movlpd	%xmm1, 7(%edx)
	mov	$15, %esi
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit2Case2OrCase3):
	movlpd	(%ecx), %xmm0
	movlpd	6(%ecx), %xmm1
	movlpd	%xmm0, (%edx)
	movlpd	%xmm1, 6(%edx)
	mov	$14, %esi
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit3Case2OrCase3):
	movlpd	(%ecx), %xmm0
	movlpd	5(%ecx), %xmm1
	movlpd	%xmm0, (%edx)
	movlpd	%xmm1, 5(%edx)
	mov	$13, %esi
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit4Case2OrCase3):
	movlpd	(%ecx), %xmm0
	movl	8(%ecx), %esi
	movlpd	%xmm0, (%edx)
	movl	%esi, 8(%edx)
	mov	$12, %esi
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit5Case2OrCase3):
	movlpd	(%ecx), %xmm0
	movl	7(%ecx), %esi
	movlpd	%xmm0, (%edx)
	movl	%esi, 7(%edx)
	mov	$11, %esi
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit6Case2OrCase3):
	movlpd	(%ecx), %xmm0
	movl	6(%ecx), %esi
	movlpd	%xmm0, (%edx)
	movl	%esi, 6(%edx)
	mov	$10, %esi
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit7Case2OrCase3):
	movlpd	(%ecx), %xmm0
	movl	5(%ecx), %esi
	movlpd	%xmm0, (%edx)
	movl	%esi, 5(%edx)
	mov	$9, %esi
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit8Case2OrCase3):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	mov	$8, %esi
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit9Case2OrCase3):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	mov	$7, %esi
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit10Case2OrCase3):
	movlpd	-1(%ecx), %xmm0
	movlpd	%xmm0, -1(%edx)
	mov	$6, %esi
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit11Case2OrCase3):
	movlpd	-2(%ecx), %xmm0
	movlpd	%xmm0, -2(%edx)
	mov	$5, %esi
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit12Case2OrCase3):
	movl	(%ecx), %esi
	movl	%esi, (%edx)
	mov	$4, %esi
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit13Case2OrCase3):
	movl	-1(%ecx), %esi
	movl	%esi, -1(%edx)
	mov	$3, %esi
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit14Case2OrCase3):
	movl	-2(%ecx), %esi
	movl	%esi, -2(%edx)
	mov	$2, %esi
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit15Case2OrCase3):
	movl	-3(%ecx), %esi
	movl	%esi, -3(%edx)
	mov	$1, %esi
	test	%eax, %eax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

L(StrncpyLeave1):
	movaps	%xmm2, %xmm3
	add	$48, %ebx
	jle	L(StrncpyExit1)
	palignr	$1, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	31(%ecx), %xmm2
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit1)
	palignr	$1, %xmm3, %xmm2
	movaps	%xmm2, 16(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit1)
	movaps	%xmm4, 32(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit1)
	movaps	%xmm5, 48(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
L(StrncpyExit1):
	lea	15(%edx, %esi), %edx
	lea	15(%ecx, %esi), %ecx
	movdqu	-16(%ecx), %xmm0
	xor	%esi, %esi
	movdqu	%xmm0, -16(%edx)
	jmp	L(CopyFrom1To16BytesCase3)

L(StrncpyLeave2):
	movaps	%xmm2, %xmm3
	add	$48, %ebx
	jle	L(StrncpyExit2)
	palignr	$2, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	30(%ecx), %xmm2
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit2)
	palignr	$2, %xmm3, %xmm2
	movaps	%xmm2, 16(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit2)
	movaps	%xmm4, 32(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit2)
	movaps	%xmm5, 48(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
L(StrncpyExit2):
	lea	14(%edx, %esi), %edx
	lea	14(%ecx, %esi), %ecx
	movdqu	-16(%ecx), %xmm0
	xor	%esi, %esi
	movdqu	%xmm0, -16(%edx)
	jmp	L(CopyFrom1To16BytesCase3)

L(StrncpyLeave3):
	movaps	%xmm2, %xmm3
	add	$48, %ebx
	jle	L(StrncpyExit3)
	palignr	$3, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	29(%ecx), %xmm2
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit3)
	palignr	$3, %xmm3, %xmm2
	movaps	%xmm2, 16(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit3)
	movaps	%xmm4, 32(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit3)
	movaps	%xmm5, 48(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
L(StrncpyExit3):
	lea	13(%edx, %esi), %edx
	lea	13(%ecx, %esi), %ecx
	movdqu	-16(%ecx), %xmm0
	xor	%esi, %esi
	movdqu	%xmm0, -16(%edx)
	jmp	L(CopyFrom1To16BytesCase3)

L(StrncpyLeave4):
	movaps	%xmm2, %xmm3
	add	$48, %ebx
	jle	L(StrncpyExit4)
	palignr	$4, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	28(%ecx), %xmm2
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit4)
	palignr	$4, %xmm3, %xmm2
	movaps	%xmm2, 16(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit4)
	movaps	%xmm4, 32(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit4)
	movaps	%xmm5, 48(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
L(StrncpyExit4):
	lea	12(%edx, %esi), %edx
	lea	12(%ecx, %esi), %ecx
	movlpd	-12(%ecx), %xmm0
	movl	-4(%ecx), %eax
	movlpd	%xmm0, -12(%edx)
	movl	%eax, -4(%edx)
	xor	%esi, %esi
	jmp	L(CopyFrom1To16BytesCase3)

L(StrncpyLeave5):
	movaps	%xmm2, %xmm3
	add	$48, %ebx
	jle	L(StrncpyExit5)
	palignr	$5, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	27(%ecx), %xmm2
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit5)
	palignr	$5, %xmm3, %xmm2
	movaps	%xmm2, 16(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit5)
	movaps	%xmm4, 32(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit5)
	movaps	%xmm5, 48(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
L(StrncpyExit5):
	lea	11(%edx, %esi), %edx
	lea	11(%ecx, %esi), %ecx
	movlpd	-11(%ecx), %xmm0
	movl	-4(%ecx), %eax
	movlpd	%xmm0, -11(%edx)
	movl	%eax, -4(%edx)
	xor	%esi, %esi
	jmp	L(CopyFrom1To16BytesCase3)

L(StrncpyLeave6):
	movaps	%xmm2, %xmm3
	add	$48, %ebx
	jle	L(StrncpyExit6)
	palignr	$6, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	26(%ecx), %xmm2
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit6)
	palignr	$6, %xmm3, %xmm2
	movaps	%xmm2, 16(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit6)
	movaps	%xmm4, 32(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit6)
	movaps	%xmm5, 48(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
L(StrncpyExit6):
	lea	10(%edx, %esi), %edx
	lea	10(%ecx, %esi), %ecx

	movlpd	-10(%ecx), %xmm0
	movw	-2(%ecx), %ax
	movlpd	%xmm0, -10(%edx)
	movw	%ax, -2(%edx)
	xor	%esi, %esi
	jmp	L(CopyFrom1To16BytesCase3)

L(StrncpyLeave7):
	movaps	%xmm2, %xmm3
	add	$48, %ebx
	jle	L(StrncpyExit7)
	palignr	$7, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	25(%ecx), %xmm2
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit7)
	palignr	$7, %xmm3, %xmm2
	movaps	%xmm2, 16(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit7)
	movaps	%xmm4, 32(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit7)
	movaps	%xmm5, 48(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
L(StrncpyExit7):
	lea	9(%edx, %esi), %edx
	lea	9(%ecx, %esi), %ecx

	movlpd	-9(%ecx), %xmm0
	movb	-1(%ecx), %ah
	movlpd	%xmm0, -9(%edx)
	movb	%ah, -1(%edx)
	xor	%esi, %esi
	jmp	L(CopyFrom1To16BytesCase3)

L(StrncpyLeave8):
	movaps	%xmm2, %xmm3
	add	$48, %ebx
	jle	L(StrncpyExit8)
	palignr	$8, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	24(%ecx), %xmm2
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit8)
	palignr	$8, %xmm3, %xmm2
	movaps	%xmm2, 16(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit8)
	movaps	%xmm4, 32(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit8)
	movaps	%xmm5, 48(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
L(StrncpyExit8):
	lea	8(%edx, %esi), %edx
	lea	8(%ecx, %esi), %ecx
	movlpd	-8(%ecx), %xmm0
	movlpd	%xmm0, -8(%edx)
	xor	%esi, %esi
	jmp	L(CopyFrom1To16BytesCase3)

L(StrncpyLeave9):
	movaps	%xmm2, %xmm3
	add	$48, %ebx
	jle	L(StrncpyExit9)
	palignr	$9, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	23(%ecx), %xmm2
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit9)
	palignr	$9, %xmm3, %xmm2
	movaps	%xmm2, 16(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit9)
	movaps	%xmm4, 32(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit9)
	movaps	%xmm5, 48(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
L(StrncpyExit9):
	lea	7(%edx, %esi), %edx
	lea	7(%ecx, %esi), %ecx

	movlpd	-8(%ecx), %xmm0
	movlpd	%xmm0, -8(%edx)
	xor	%esi, %esi
	jmp	L(CopyFrom1To16BytesCase3)

L(StrncpyLeave10):
	movaps	%xmm2, %xmm3
	add	$48, %ebx
	jle	L(StrncpyExit10)
	palignr	$10, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	22(%ecx), %xmm2
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit10)
	palignr	$10, %xmm3, %xmm2
	movaps	%xmm2, 16(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit10)
	movaps	%xmm4, 32(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit10)
	movaps	%xmm5, 48(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
L(StrncpyExit10):
	lea	6(%edx, %esi), %edx
	lea	6(%ecx, %esi), %ecx

	movlpd	-8(%ecx), %xmm0
	movlpd	%xmm0, -8(%edx)
	xor	%esi, %esi
	jmp	L(CopyFrom1To16BytesCase3)

L(StrncpyLeave11):
	movaps	%xmm2, %xmm3
	add	$48, %ebx
	jle	L(StrncpyExit11)
	palignr	$11, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	21(%ecx), %xmm2
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit11)
	palignr	$11, %xmm3, %xmm2
	movaps	%xmm2, 16(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit11)
	movaps	%xmm4, 32(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit11)
	movaps	%xmm5, 48(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
L(StrncpyExit11):
	lea	5(%edx, %esi), %edx
	lea	5(%ecx, %esi), %ecx
	movl	-5(%ecx), %esi
	movb	-1(%ecx), %ah
	movl	%esi, -5(%edx)
	movb	%ah, -1(%edx)
	xor	%esi, %esi
	jmp	L(CopyFrom1To16BytesCase3)

L(StrncpyLeave12):
	movaps	%xmm2, %xmm3
	add	$48, %ebx
	jle	L(StrncpyExit12)
	palignr	$12, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	20(%ecx), %xmm2
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit12)
	palignr	$12, %xmm3, %xmm2
	movaps	%xmm2, 16(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit12)
	movaps	%xmm4, 32(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit12)
	movaps	%xmm5, 48(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
L(StrncpyExit12):
	lea	4(%edx, %esi), %edx
	lea	4(%ecx, %esi), %ecx
	movl	-4(%ecx), %eax
	movl	%eax, -4(%edx)
	xor	%esi, %esi
	jmp	L(CopyFrom1To16BytesCase3)

L(StrncpyLeave13):
	movaps	%xmm2, %xmm3
	add	$48, %ebx
	jle	L(StrncpyExit13)
	palignr	$13, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	19(%ecx), %xmm2
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit13)
	palignr	$13, %xmm3, %xmm2
	movaps	%xmm2, 16(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit13)
	movaps	%xmm4, 32(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit13)
	movaps	%xmm5, 48(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
L(StrncpyExit13):
	lea	3(%edx, %esi), %edx
	lea	3(%ecx, %esi), %ecx

	movl	-4(%ecx), %eax
	movl	%eax, -4(%edx)
	xor	%esi, %esi
	jmp	L(CopyFrom1To16BytesCase3)

L(StrncpyLeave14):
	movaps	%xmm2, %xmm3
	add	$48, %ebx
	jle	L(StrncpyExit14)
	palignr	$14, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	18(%ecx), %xmm2
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit14)
	palignr	$14, %xmm3, %xmm2
	movaps	%xmm2, 16(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit14)
	movaps	%xmm4, 32(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit14)
	movaps	%xmm5, 48(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
L(StrncpyExit14):
	lea	2(%edx, %esi), %edx
	lea	2(%ecx, %esi), %ecx
	movw	-2(%ecx), %ax
	movw	%ax, -2(%edx)
	xor	%esi, %esi
	jmp	L(CopyFrom1To16BytesCase3)

L(StrncpyLeave15):
	movaps	%xmm2, %xmm3
	add	$48, %ebx
	jle	L(StrncpyExit15)
	palignr	$15, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	17(%ecx), %xmm2
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit15)
	palignr	$15, %xmm3, %xmm2
	movaps	%xmm2, 16(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit15)
	movaps	%xmm4, 32(%edx)
	lea	16(%esi), %esi
	sub	$16, %ebx
	jbe	L(StrncpyExit15)
	movaps	%xmm5, 48(%edx)
	lea	16(%esi), %esi
	lea	-16(%ebx), %ebx
L(StrncpyExit15):
	lea	1(%edx, %esi), %edx
	lea	1(%ecx, %esi), %ecx
	movb	-1(%ecx), %ah
	movb	%ah, -1(%edx)
	xor	%esi, %esi
	jmp	L(CopyFrom1To16BytesCase3)
# endif

# ifndef USE_AS_STRCAT
#  ifdef USE_AS_STRNCPY
	CFI_POP (%esi)
	CFI_POP (%edi)

	.p2align 4
L(ExitTail0):
	movl	%edx, %eax
	RETURN

	.p2align 4
L(StrncpyExit15Bytes):
	cmp	$12, %ebx
	jbe	L(StrncpyExit12Bytes)
	cmpb	$0, 8(%ecx)
	jz	L(ExitTail9)
	cmpb	$0, 9(%ecx)
	jz	L(ExitTail10)
	cmpb	$0, 10(%ecx)
	jz	L(ExitTail11)
	cmpb	$0, 11(%ecx)
	jz	L(ExitTail12)
	cmp	$13, %ebx
	je	L(ExitTail13)
	cmpb	$0, 12(%ecx)
	jz	L(ExitTail13)
	cmp	$14, %ebx
	je	L(ExitTail14)
	cmpb	$0, 13(%ecx)
	jz	L(ExitTail14)
	movlpd	(%ecx), %xmm0
	movlpd	7(%ecx), %xmm1
	movlpd	%xmm0, (%edx)
	movlpd	%xmm1, 7(%edx)
#   ifdef USE_AS_STPCPY
	lea	14(%edx), %eax
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   else
	movl	%edx, %eax
#   endif
	RETURN

	.p2align 4
L(StrncpyExit12Bytes):
	cmp	$9, %ebx
	je	L(ExitTail9)
	cmpb	$0, 8(%ecx)
	jz	L(ExitTail9)
	cmp	$10, %ebx
	je	L(ExitTail10)
	cmpb	$0, 9(%ecx)
	jz	L(ExitTail10)
	cmp	$11, %ebx
	je	L(ExitTail11)
	cmpb	$0, 10(%ecx)
	jz	L(ExitTail11)
	movlpd	(%ecx), %xmm0
	movl	8(%ecx), %eax
	movlpd	%xmm0, (%edx)
	movl	%eax, 8(%edx)
	SAVE_RESULT_TAIL (11)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
	RETURN

	.p2align 4
L(StrncpyExit8Bytes):
	cmp	$4, %ebx
	jbe	L(StrncpyExit4Bytes)
	cmpb	$0, (%ecx)
	jz	L(ExitTail1)
	cmpb	$0, 1(%ecx)
	jz	L(ExitTail2)
	cmpb	$0, 2(%ecx)
	jz	L(ExitTail3)
	cmpb	$0, 3(%ecx)
	jz	L(ExitTail4)

	cmp	$5, %ebx
	je	L(ExitTail5)
	cmpb	$0, 4(%ecx)
	jz	L(ExitTail5)
	cmp	$6, %ebx
	je	L(ExitTail6)
	cmpb	$0, 5(%ecx)
	jz	L(ExitTail6)
	cmp	$7, %ebx
	je	L(ExitTail7)
	cmpb	$0, 6(%ecx)
	jz	L(ExitTail7)
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
#   ifdef USE_AS_STPCPY
	lea	7(%edx), %eax
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   else
	movl	%edx, %eax
#   endif
	RETURN

	.p2align 4
L(StrncpyExit4Bytes):
	test	%ebx, %ebx
	jz	L(ExitTail0)
	cmp	$1, %ebx
	je	L(ExitTail1)
	cmpb	$0, (%ecx)
	jz	L(ExitTail1)
	cmp	$2, %ebx
	je	L(ExitTail2)
	cmpb	$0, 1(%ecx)
	jz	L(ExitTail2)
	cmp	$3, %ebx
	je	L(ExitTail3)
	cmpb	$0, 2(%ecx)
	jz	L(ExitTail3)
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	SAVE_RESULT_TAIL (3)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%eax)
	sbb	$-1, %eax
#   endif
	RETURN
#  endif

END (STRCPY)
# endif
#endif

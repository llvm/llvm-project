/* wcscpy with SSSE3
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
# define RETURN	POP (%edi); ret; CFI_PUSH (%edi)
# define STR1	PARMS
# define STR2	STR1+4
# define LEN	STR2+4

	atom_text_section
ENTRY (__wcscpy_ssse3)
	mov	STR1(%esp), %edx
	mov	STR2(%esp), %ecx

	cmp	$0, (%ecx)
	jz	L(ExitTail4)
	cmp	$0, 4(%ecx)
	jz	L(ExitTail8)
	cmp	$0, 8(%ecx)
	jz	L(ExitTail12)
	cmp	$0, 12(%ecx)
	jz	L(ExitTail16)

	PUSH	(%edi)
	mov	%edx, %edi
	PUSH	(%esi)
	lea	16(%ecx), %esi

	and	$-16, %esi

	pxor	%xmm0, %xmm0
	pcmpeqd	(%esi), %xmm0
	movdqu	(%ecx), %xmm1
	movdqu	%xmm1, (%edx)

	pmovmskb %xmm0, %eax
	sub	%ecx, %esi

	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	mov	%edx, %eax
	lea	16(%edx), %edx
	and	$-16, %edx
	sub	%edx, %eax

	sub	%eax, %ecx
	mov	%ecx, %eax
	and	$0xf, %eax
	mov	$0, %esi

	jz	L(Align16Both)
	cmp	$4, %eax
	je	L(Shl4)
	cmp	$8, %eax
	je	L(Shl8)
	jmp	L(Shl12)

L(Align16Both):
	movaps	(%ecx), %xmm1
	movaps	16(%ecx), %xmm2
	movaps	%xmm1, (%edx)
	pcmpeqd	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	lea	16(%esi), %esi

	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%ecx, %esi), %xmm3
	movaps	%xmm2, (%edx, %esi)
	pcmpeqd	%xmm3, %xmm0
	pmovmskb %xmm0, %eax
	lea	16(%esi), %esi

	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%ecx, %esi), %xmm4
	movaps	%xmm3, (%edx, %esi)
	pcmpeqd	%xmm4, %xmm0
	pmovmskb %xmm0, %eax
	lea	16(%esi), %esi

	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%ecx, %esi), %xmm1
	movaps	%xmm4, (%edx, %esi)
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	lea	16(%esi), %esi

	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%ecx, %esi), %xmm2
	movaps	%xmm1, (%edx, %esi)
	pcmpeqd	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	lea	16(%esi), %esi

	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%ecx, %esi), %xmm3
	movaps	%xmm2, (%edx, %esi)
	pcmpeqd	%xmm3, %xmm0
	pmovmskb %xmm0, %eax
	lea	16(%esi), %esi

	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	movaps	%xmm3, (%edx, %esi)
	mov	%ecx, %eax
	lea	16(%ecx, %esi), %ecx
	and	$-0x40, %ecx
	sub	%ecx, %eax
	sub	%eax, %edx

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
	pcmpeqd	%xmm0, %xmm3
	lea	64(%ecx), %ecx
	pmovmskb %xmm3, %eax

	test	%eax, %eax
	jnz	L(Aligned64Leave)
	movaps	%xmm4, -64(%edx)
	movaps	%xmm5, -48(%edx)
	movaps	%xmm6, -32(%edx)
	movaps	%xmm7, -16(%edx)
	jmp	L(Aligned64Loop)

L(Aligned64Leave):
	pcmpeqd	%xmm4, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqd	%xmm5, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm4, -64(%edx)
	test	%eax, %eax
	lea	16(%esi), %esi
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqd	%xmm6, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm5, -48(%edx)
	test	%eax, %eax
	lea	16(%esi), %esi
	jnz	L(CopyFrom1To16Bytes)

	movaps	%xmm6, -32(%edx)
	pcmpeqd	%xmm7, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	lea	16(%esi), %esi
	jnz	L(CopyFrom1To16Bytes)

	mov	$-0x40, %esi
	movaps	%xmm7, -16(%edx)
	jmp	L(Aligned64Loop)

	.p2align 4
L(Shl4):
	movaps	-4(%ecx), %xmm1
	movaps	12(%ecx), %xmm2
L(Shl4Start):
	pcmpeqd	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3

	test	%eax, %eax
	jnz	L(Shl4LoopExit)

	palignr	$4, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	28(%ecx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm1

	test	%eax, %eax
	jnz	L(Shl4LoopExit)

	palignr	$4, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	movaps	28(%ecx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3

	test	%eax, %eax
	jnz	L(Shl4LoopExit)

	palignr	$4, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	28(%ecx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx

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
	pcmpeqd	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$4, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$4, %xmm3, %xmm4
	jnz	L(Shl4Start)

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
	POP	(%esi)
	add	$12, %edx
	add	$12, %ecx
	test	%al, %al
	jz	L(ExitHigh)
	test	$0x01, %al
	jnz	L(Exit4)
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	movl	%edi, %eax
	RETURN

	CFI_PUSH	(%esi)

	.p2align 4
L(Shl8):
	movaps	-8(%ecx), %xmm1
	movaps	8(%ecx), %xmm2
L(Shl8Start):
	pcmpeqd	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3

	test	%eax, %eax
	jnz	L(Shl8LoopExit)

	palignr	$8, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	24(%ecx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm1

	test	%eax, %eax
	jnz	L(Shl8LoopExit)

	palignr	$8, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	movaps	24(%ecx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3

	test	%eax, %eax
	jnz	L(Shl8LoopExit)

	palignr	$8, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	24(%ecx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx

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
	pcmpeqd	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$8, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$8, %xmm3, %xmm4
	jnz	L(Shl8Start)

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
	POP	(%esi)
	add	$8, %edx
	add	$8, %ecx
	test	%al, %al
	jz	L(ExitHigh)
	test	$0x01, %al
	jnz	L(Exit4)
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	movl	%edi, %eax
	RETURN

	CFI_PUSH	(%esi)

	.p2align 4
L(Shl12):
	movaps	-12(%ecx), %xmm1
	movaps	4(%ecx), %xmm2
L(Shl12Start):
	pcmpeqd	%xmm2, %xmm0
	pmovmskb %xmm0, %eax
	movaps	%xmm2, %xmm3

	test	%eax, %eax
	jnz	L(Shl12LoopExit)

	palignr	$12, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	20(%ecx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm1

	test	%eax, %eax
	jnz	L(Shl12LoopExit)

	palignr	$12, %xmm3, %xmm2
	movaps	%xmm2, (%edx)
	movaps	20(%ecx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx
	movaps	%xmm2, %xmm3

	test	%eax, %eax
	jnz	L(Shl12LoopExit)

	palignr	$12, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	20(%ecx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%edx), %edx
	pmovmskb %xmm0, %eax
	lea	16(%ecx), %ecx

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
	pcmpeqd	%xmm0, %xmm7
	pmovmskb %xmm7, %eax
	movaps	%xmm5, %xmm7
	palignr	$12, %xmm4, %xmm5
	test	%eax, %eax
	palignr	$12, %xmm3, %xmm4
	jnz	L(Shl12Start)

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

	.p2align 4
L(CopyFrom1To16Bytes):
	add	%esi, %edx
	add	%esi, %ecx

	POP	(%esi)
	test	%al, %al
	jz	L(ExitHigh)
	test	$0x01, %al
	jnz	L(Exit4)
L(Exit8):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	movl	%edi, %eax
	RETURN

	.p2align 4
L(ExitHigh):
	test	$0x01, %ah
	jnz	L(Exit12)
L(Exit16):
	movdqu	(%ecx), %xmm0
	movdqu	%xmm0, (%edx)
	movl	%edi, %eax
	RETURN

	.p2align 4
L(Exit4):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	%edi, %eax
	RETURN

	.p2align 4
L(Exit12):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	movl	8(%ecx), %eax
	movl	%eax, 8(%edx)
	movl	%edi, %eax
	RETURN

CFI_POP	(%edi)

	.p2align 4
L(ExitTail4):
	movl	(%ecx), %eax
	movl	%eax, (%edx)
	movl	%edx, %eax
	ret

	.p2align 4
L(ExitTail8):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	movl	%edx, %eax
	ret

	.p2align 4
L(ExitTail12):
	movlpd	(%ecx), %xmm0
	movlpd	%xmm0, (%edx)
	movl	8(%ecx), %eax
	movl	%eax, 8(%edx)
	movl	%edx, %eax
	ret

	.p2align 4
L(ExitTail16):
	movdqu	(%ecx), %xmm0
	movdqu	%xmm0, (%edx)
	movl	%edx, %eax
	ret

END (__wcscpy_ssse3)
#endif

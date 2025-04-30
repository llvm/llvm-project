/* strcat(dest, src) -- Append SRC on the end of DEST.
   Optimized for x86-64.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 2002.

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

#include <sysdep.h>
#include "asm-syntax.h"

/* Will be removed when new strcpy implementation gets merged.  */

	.text
ENTRY (strcat)
	movq %rdi, %rcx		/* Dest. register. */
	andl $7, %ecx		/* mask alignment bits */
	movq %rdi, %rax		/* Duplicate destination pointer.  */
	movq $0xfefefefefefefeff,%r8

	/* First step: Find end of destination.  */
	jz 4f			/* aligned => start loop */

	neg %ecx		/* We need to align to 8 bytes.  */
	addl $8,%ecx
	/* Search the first bytes directly.  */
0:	cmpb $0x0,(%rax)	/* is byte NUL? */
	je 2f			/* yes => start copy */
	incq %rax		/* increment pointer */
	decl %ecx
	jnz 0b



	/* Now the source is aligned.  Scan for NUL byte.  */
	.p2align 4
4:
	/* First unroll.  */
	movq (%rax), %rcx	/* get double word (= 8 bytes) in question */
	addq $8,%rax		/* adjust pointer for next word */
	movq %r8, %rdx		/* magic value */
	addq %rcx, %rdx		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc 3f			/* highest byte is NUL => return pointer */
	xorq %rcx, %rdx		/* (word+magic)^word */
	orq %r8, %rdx		/* set all non-carry bits */
	incq %rdx		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz 3f			/* found NUL => return pointer */

	/* Second unroll.  */
	movq (%rax), %rcx	/* get double word (= 8 bytes) in question */
	addq $8,%rax		/* adjust pointer for next word */
	movq %r8, %rdx		/* magic value */
	addq %rcx, %rdx		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc 3f			/* highest byte is NUL => return pointer */
	xorq %rcx, %rdx		/* (word+magic)^word */
	orq %r8, %rdx		/* set all non-carry bits */
	incq %rdx		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz 3f			/* found NUL => return pointer */

	/* Third unroll.  */
	movq (%rax), %rcx	/* get double word (= 8 bytes) in question */
	addq $8,%rax		/* adjust pointer for next word */
	movq %r8, %rdx		/* magic value */
	addq %rcx, %rdx		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc 3f			/* highest byte is NUL => return pointer */
	xorq %rcx, %rdx		/* (word+magic)^word */
	orq %r8, %rdx		/* set all non-carry bits */
	incq %rdx		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz 3f			/* found NUL => return pointer */

	/* Fourth unroll.  */
	movq (%rax), %rcx	/* get double word (= 8 bytes) in question */
	addq $8,%rax		/* adjust pointer for next word */
	movq %r8, %rdx		/* magic value */
	addq %rcx, %rdx		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc 3f			/* highest byte is NUL => return pointer */
	xorq %rcx, %rdx		/* (word+magic)^word */
	orq %r8, %rdx		/* set all non-carry bits */
	incq %rdx		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jz 4b			/* no NUL found => continue loop */

	.p2align 4		/* Align, it's a jump target.  */
3:	subq $8,%rax		/* correct pointer increment.  */

	testb %cl, %cl		/* is first byte NUL? */
	jz 2f			/* yes => return */
	incq %rax		/* increment pointer */

	testb %ch, %ch		/* is second byte NUL? */
	jz 2f			/* yes => return */
	incq %rax		/* increment pointer */

	testl $0x00ff0000, %ecx /* is third byte NUL? */
	jz 2f			/* yes => return pointer */
	incq %rax		/* increment pointer */

	testl $0xff000000, %ecx /* is fourth byte NUL? */
	jz 2f			/* yes => return pointer */
	incq %rax		/* increment pointer */

	shrq $32, %rcx		/* look at other half.  */

	testb %cl, %cl		/* is first byte NUL? */
	jz 2f			/* yes => return */
	incq %rax		/* increment pointer */

	testb %ch, %ch		/* is second byte NUL? */
	jz 2f			/* yes => return */
	incq %rax		/* increment pointer */

	testl $0xff0000, %ecx	/* is third byte NUL? */
	jz 2f			/* yes => return pointer */
	incq %rax		/* increment pointer */

2:
	/* Second step: Copy source to destination.  */

	movq	%rsi, %rcx	/* duplicate  */
	andl	$7,%ecx		/* mask alignment bits */
	movq	%rax, %rdx	/* move around */
	jz	22f		/* aligned => start loop */

	neg	%ecx		/* align to 8 bytes.  */
	addl	$8, %ecx
	/* Align the source pointer.  */
21:
	movb	(%rsi), %al	/* Fetch a byte */
	testb	%al, %al	/* Is it NUL? */
	movb	%al, (%rdx)	/* Store it */
	jz	24f		/* If it was NUL, done! */
	incq	%rsi
	incq	%rdx
	decl	%ecx
	jnz	21b

	/* Now the sources is aligned.  Unfortunatly we cannot force
	   to have both source and destination aligned, so ignore the
	   alignment of the destination.  */
	.p2align 4
22:
	/* 1st unroll.  */
	movq	(%rsi), %rax	/* Read double word (8 bytes).  */
	addq	$8, %rsi	/* Adjust pointer for next word.  */
	movq	%rax, %r9	/* Save a copy for NUL finding.  */
	addq	%r8, %r9	/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc	23f		/* highest byte is NUL => return pointer */
	xorq	%rax, %r9	/* (word+magic)^word */
	orq	%r8, %r9	/* set all non-carry bits */
	incq	%r9		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */

	jnz	23f		/* found NUL => return pointer */

	movq	%rax, (%rdx)	/* Write value to destination.  */
	addq	$8, %rdx	/* Adjust pointer.  */

	/* 2nd unroll.  */
	movq	(%rsi), %rax	/* Read double word (8 bytes).  */
	addq	$8, %rsi	/* Adjust pointer for next word.  */
	movq	%rax, %r9	/* Save a copy for NUL finding.  */
	addq	%r8, %r9	/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc	23f		/* highest byte is NUL => return pointer */
	xorq	%rax, %r9	/* (word+magic)^word */
	orq	%r8, %r9	/* set all non-carry bits */
	incq	%r9		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */

	jnz	23f		/* found NUL => return pointer */

	movq	%rax, (%rdx)	/* Write value to destination.  */
	addq	$8, %rdx	/* Adjust pointer.  */

	/* 3rd unroll.  */
	movq	(%rsi), %rax	/* Read double word (8 bytes).  */
	addq	$8, %rsi	/* Adjust pointer for next word.  */
	movq	%rax, %r9	/* Save a copy for NUL finding.  */
	addq	%r8, %r9	/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc	23f		/* highest byte is NUL => return pointer */
	xorq	%rax, %r9	/* (word+magic)^word */
	orq	%r8, %r9	/* set all non-carry bits */
	incq	%r9		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */

	jnz	23f		/* found NUL => return pointer */

	movq	%rax, (%rdx)	/* Write value to destination.  */
	addq	$8, %rdx	/* Adjust pointer.  */

	/* 4th unroll.  */
	movq	(%rsi), %rax	/* Read double word (8 bytes).  */
	addq	$8, %rsi	/* Adjust pointer for next word.  */
	movq	%rax, %r9	/* Save a copy for NUL finding.  */
	addq	%r8, %r9	/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc	23f		/* highest byte is NUL => return pointer */
	xorq	%rax, %r9	/* (word+magic)^word */
	orq	%r8, %r9	/* set all non-carry bits */
	incq	%r9		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */

	jnz	23f		/* found NUL => return pointer */

	movq	%rax, (%rdx)	/* Write value to destination.  */
	addq	$8, %rdx	/* Adjust pointer.  */
	jmp	22b		/* Next iteration.  */

	/* Do the last few bytes. %rax contains the value to write.
	   The loop is unrolled twice.  */
	.p2align 4
23:
	movb	%al, (%rdx)	/* 1st byte.  */
	testb	%al, %al	/* Is it NUL.  */
	jz	24f		/* yes, finish.  */
	incq	%rdx		/* Increment destination.  */
	movb	%ah, (%rdx)	/* 2nd byte.  */
	testb	%ah, %ah	/* Is it NUL?.  */
	jz	24f		/* yes, finish.  */
	incq	%rdx		/* Increment destination.  */
	shrq	$16, %rax	/* Shift...  */
	jmp	23b		/* and look at next two bytes in %rax.  */


24:
	movq	%rdi, %rax	/* Source is return value.  */
	retq
END (strcat)
libc_hidden_builtin_def (strcat)

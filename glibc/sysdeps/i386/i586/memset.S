/* memset/bzero -- set memory area to CH/0
   Highly optimized version for ix86, x>=5.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Torbjorn Granlund, <tege@matematik.su.se>

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

#define PARMS	4+4	/* space for 1 saved reg */
#define RTN	PARMS
#define DEST	RTN
#ifdef USE_AS_BZERO
# define LEN	DEST+4
#else
# define CHR	DEST+4
# define LEN	CHR+4
#endif

        .text
#if defined SHARED && IS_IN (libc) && !defined USE_AS_BZERO
ENTRY (__memset_chk)
	movl	12(%esp), %eax
	cmpl	%eax, 16(%esp)
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END (__memset_chk)
#endif
ENTRY (memset)

	pushl	%edi
	cfi_adjust_cfa_offset (4)

	movl	DEST(%esp), %edi
	cfi_rel_offset (edi, 0)
	movl	LEN(%esp), %edx
#ifdef USE_AS_BZERO
	xorl	%eax, %eax	/* we fill with 0 */
#else
	movb	CHR(%esp), %al
	movb	%al, %ah
	movl	%eax, %ecx
	shll	$16, %eax
	movw	%cx, %ax
#endif
	cld

/* If less than 36 bytes to write, skip tricky code (it wouldn't work).  */
	cmpl	$36, %edx
	movl	%edx, %ecx	/* needed when branch is taken! */
	jl	L(2)

/* First write 0-3 bytes to make the pointer 32-bit aligned.  */
	movl	%edi, %ecx	/* Copy ptr to ecx... */
	negl	%ecx		/* ...and negate that and... */
	andl	$3, %ecx	/* ...mask to get byte count.  */
	subl	%ecx, %edx	/* adjust global byte count */
	rep
	stosb

	subl	$32, %edx	/* offset count for unrolled loop */
	movl	(%edi), %ecx	/* Fetch destination cache line */

	.align	2, 0x90		/* supply 0x90 for broken assemblers */
L(1):	movl	28(%edi), %ecx	/* allocate cache line for destination */
	subl	$32, %edx	/* decr loop count */
	movl	%eax, 0(%edi)	/* store words pairwise */
	movl	%eax, 4(%edi)
	movl	%eax, 8(%edi)
	movl	%eax, 12(%edi)
	movl	%eax, 16(%edi)
	movl	%eax, 20(%edi)
	movl	%eax, 24(%edi)
	movl	%eax, 28(%edi)
	leal	32(%edi), %edi	/* update destination pointer */
	jge	L(1)

	leal	32(%edx), %ecx	/* reset offset count */

/* Write last 0-7 full 32-bit words (up to 8 words if loop was skipped).  */
L(2):	shrl	$2, %ecx	/* convert byte count to longword count */
	rep
	stosl

/* Finally write the last 0-3 bytes.  */
	movl	%edx, %ecx
	andl	$3, %ecx
	rep
	stosb

#ifndef USE_AS_BZERO
	/* Load result (only if used as memset).  */
	movl DEST(%esp), %eax	/* start address of destination is result */
#endif
	popl	%edi
	cfi_adjust_cfa_offset (-4)
	cfi_restore (edi)

	ret
END (memset)
libc_hidden_builtin_def (memset)

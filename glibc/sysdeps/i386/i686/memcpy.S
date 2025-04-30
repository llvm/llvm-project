/* Copy memory block and return pointer to beginning of destination block
   For Intel 80x86, x>=6.
   This file is part of the GNU C Library.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1999.

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

#define PARMS	4		/* no space for saved regs */
#define RTN	PARMS
#define DEST	RTN
#define SRC	DEST+4
#define LEN	SRC+4

	.text
#if defined PIC && IS_IN (libc)
ENTRY_CHK (__memcpy_chk)
	movl	12(%esp), %eax
	cmpl	%eax, 16(%esp)
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END_CHK (__memcpy_chk)
#endif
ENTRY (memcpy)

	movl	%edi, %eax
	movl	DEST(%esp), %edi
	movl	%esi, %edx
	movl	SRC(%esp), %esi

	movl	%edi, %ecx
	xorl	%esi, %ecx
	andl	$3, %ecx
	movl	LEN(%esp), %ecx
	cld
	jne	.Lunaligned

	cmpl	$3, %ecx
	jbe	.Lunaligned

	testl	$3, %esi
	je	1f
	movsb
	decl	%ecx
	testl	$3, %esi
	je	1f
	movsb
	decl	%ecx
	testl	$3, %esi
	je	1f
	movsb
	decl	%ecx
1:	pushl	%eax
	movl	%ecx, %eax
	shrl	$2, %ecx
	andl	$3, %eax
	rep
	movsl
	movl	%eax, %ecx
	rep
	movsb
	popl	%eax

.Lend:	movl	%eax, %edi
	movl	%edx, %esi
	movl	DEST(%esp), %eax

	ret

	/* When we come here the pointers do not have the same
	   alignment or the length is too short.  No need to optimize for
	   aligned memory accesses. */
.Lunaligned:
	shrl	$1, %ecx
	jnc	1f
	movsb
1:	shrl	$1, %ecx
	jnc	2f
	movsw
2:	rep
	movsl
	jmp	.Lend
END (memcpy)
libc_hidden_builtin_def (memcpy)

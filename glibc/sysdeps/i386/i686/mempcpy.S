/* Copy memory block and return pointer to following byte.
   For Intel 80x86, x>=6.
   This file is part of the GNU C Library.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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
ENTRY_CHK (__mempcpy_chk)
	movl	12(%esp), %eax
	cmpl	%eax, 16(%esp)
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END_CHK (__mempcpy_chk)
#endif
ENTRY (__mempcpy)

	movl	LEN(%esp), %ecx
	movl	%edi, %eax
	cfi_register (edi, eax)
	movl	DEST(%esp), %edi
	movl	%esi, %edx
	cfi_register (esi, edx)
	movl	SRC(%esp), %esi
	cld
	shrl	$1, %ecx
	jnc	1f
	movsb
1:	shrl	$1, %ecx
	jnc	2f
	movsw
2:	rep
	movsl
	xchgl	%edi, %eax
	cfi_restore (edi)
	movl	%edx, %esi
	cfi_restore (esi)

	ret
END (__mempcpy)
libc_hidden_def (__mempcpy)
weak_alias (__mempcpy, mempcpy)
libc_hidden_builtin_def (mempcpy)

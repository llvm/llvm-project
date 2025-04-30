/* Highly optimized version for i586.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#define PARMS	4+8	/* space for 2 saved regs */
#define RTN	PARMS
#define DEST	RTN
#define SRC	DEST+4
#define LEN	SRC+4

        .text
#if defined PIC && IS_IN (libc)
ENTRY (__memcpy_chk)
	movl	12(%esp), %eax
	cmpl	%eax, 16(%esp)
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END (__memcpy_chk)
#endif
ENTRY (memcpy)

	pushl	%edi
	cfi_adjust_cfa_offset (4)
	pushl	%esi
	cfi_adjust_cfa_offset (4)

	movl	DEST(%esp), %edi
	cfi_rel_offset (edi, 4)
	movl	SRC(%esp), %esi
	cfi_rel_offset (esi, 0)
	movl	LEN(%esp), %ecx
	movl	%edi, %eax

	/* We need this in any case.  */
	cld

	/* Cutoff for the big loop is a size of 32 bytes since otherwise
	   the loop will never be entered.  */
	cmpl	$32, %ecx
	jbe	L(1)

	negl	%eax
	andl	$3, %eax
	subl	%eax, %ecx
	xchgl	%eax, %ecx

	rep; movsb

	movl	%eax, %ecx
	subl	$32, %ecx
	js	L(2)

	/* Read ahead to make sure we write in the cache since the stupid
	   i586 designers haven't implemented read-on-write-miss.  */
	movl	(%edi), %eax
L(3):	movl	28(%edi), %edx

	/* Now correct the loop counter.  Please note that in the following
	   code the flags are not changed anymore.  */
	subl	$32, %ecx

	movl	(%esi), %eax
	movl	4(%esi), %edx
	movl	%eax, (%edi)
	movl	%edx, 4(%edi)
	movl	8(%esi), %eax
	movl	12(%esi), %edx
	movl	%eax, 8(%edi)
	movl	%edx, 12(%edi)
	movl	16(%esi), %eax
	movl	20(%esi), %edx
	movl	%eax, 16(%edi)
	movl	%edx, 20(%edi)
	movl	24(%esi), %eax
	movl	28(%esi), %edx
	movl	%eax, 24(%edi)
	movl	%edx, 28(%edi)

	leal	32(%esi), %esi
	leal	32(%edi), %edi

	jns	L(3)

	/* Correct extra loop counter modification.  */
L(2):	addl	$32, %ecx
#ifndef USE_AS_MEMPCPY
	movl	DEST(%esp), %eax
#endif

L(1):	rep; movsb

#ifdef USE_AS_MEMPCPY
	movl	%edi, %eax
#endif

	popl	%esi
	cfi_adjust_cfa_offset (-4)
	cfi_restore (esi)
	popl	%edi
	cfi_adjust_cfa_offset (-4)
	cfi_restore (edi)

	ret
END (memcpy)
#ifndef USE_AS_MEMPCPY
libc_hidden_builtin_def (memcpy)
#endif

/* strcpy/stpcpy implementation for i586.
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

#define PARMS	4+12	/* space for 3 saved regs */
#define RTN	PARMS
#define DEST	RTN
#define SRC	DEST+4

#ifndef USE_AS_STPCPY
# define STRCPY strcpy
#endif

#define magic 0xfefefeff

	.text
ENTRY (STRCPY)

	pushl	%edi
	cfi_adjust_cfa_offset (4)
	pushl	%esi
	cfi_adjust_cfa_offset (4)
	pushl	%ebx
	cfi_adjust_cfa_offset (4)

	movl	DEST(%esp), %edi
	cfi_rel_offset (edi, 8)
	movl	SRC(%esp), %esi
	cfi_rel_offset (esi, 4)

	xorl	%eax, %eax
	leal	-1(%esi), %ecx

	movl	$magic, %ebx
	cfi_rel_offset (ebx, 0)
	andl	$3, %ecx

	cmpb	$2, %cl
	je	L(Src2)
	ja	L(Src3)
	cmpb	$1, %cl
	je	L(Src1)

L(Src0):
	orb	(%esi), %al
	jz	L(end)
	stosb
	xorl	%eax, %eax
	incl	%esi

L(Src1):
	orb	(%esi), %al
	jz	L(end)
	stosb
	xorl	%eax, %eax
	incl	%esi

L(Src2):
	orb	(%esi), %al
	jz	L(end)
	stosb
	xorl	%eax, %eax
	incl	%esi

L(Src3):
	movl	(%esi), %ecx
	leal	4(%esi),%esi

	subl	%ecx, %eax
	addl	%ebx, %ecx

	decl	%eax
	jnc	L(3)

	movl	%ecx, %edx
	xorl	%ecx, %eax

	subl	%ebx, %edx
	andl	$~magic, %eax

	jne	L(4)

	movl	%edx, (%edi)
	leal	4(%edi),%edi

	jmp	L(Src3)

L(3):	movl	%ecx, %edx

	subl	%ebx, %edx

L(4):	movb	%dl, (%edi)
	testb	%dl, %dl

	movl	%edx, %eax
	jz	L(end2)

	shrl	$16, %eax
	movb	%dh, 1(%edi)
#ifdef USE_AS_STPCPY
	addl	$1, %edi
#endif

	cmpb	$0, %dh
	jz	L(end2)

#ifdef USE_AS_STPCPY
	movb	%al, 1(%edi)
	addl	$1, %edi

	cmpb	$0, %al
	jz	L(end2)

	addl	$1, %edi
#else
	movb	%al, 2(%edi)
	testb	%al, %al

	leal	3(%edi), %edi
	jz	L(end2)
#endif

L(end):	movb	%ah, (%edi)

L(end2):
#ifdef USE_AS_STPCPY
	movl	%edi, %eax
#else
	movl	DEST(%esp), %eax
#endif
	popl	%ebx
	cfi_adjust_cfa_offset (-4)
	cfi_restore (ebx)
	popl	%esi
	cfi_adjust_cfa_offset (-4)
	cfi_restore (esi)
	popl	%edi
	cfi_adjust_cfa_offset (-4)
	cfi_restore (edi)

	ret
END (STRCPY)
#ifndef USE_AS_STPCPY
libc_hidden_builtin_def (strcpy)
#endif

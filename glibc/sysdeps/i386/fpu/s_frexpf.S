/* ix87 specific frexp implementation for float.
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

#include <machine/asm.h>
#include <libm-alias-float.h>

	.section .rodata

	.align ALIGNARG(4)
	.type two25,@object
two25:	.byte 0, 0, 0, 0x4c
	ASM_SIZE_DIRECTIVE(two25)

#ifdef PIC
#define MO(op) op##@GOTOFF(%edx)
#else
#define MO(op) op
#endif

#define PARMS	4		/* no space for saved regs */
#define VAL	PARMS
#define EXPP	VAL+4

	.text
ENTRY (__frexpf)

	movl	VAL(%esp), %eax
	xorl	%ecx, %ecx
	movl	%eax, %edx
	andl	$0x7fffffff, %eax
	jz	1f
	cmpl	$0x7f800000, %eax
	jae	1f

	cmpl	$0x00800000, %eax
	jae	2f

	flds	VAL(%esp)
#ifdef	PIC
	LOAD_PIC_REG (dx)
#endif
	fmuls	MO(two25)
	movl	$-25, %ecx
	fstps	VAL(%esp)
	fwait
	movl	VAL(%esp), %eax
	movl	%eax, %edx
	andl	$0x7fffffff, %eax

2:	shrl	$23, %eax
	andl	$0x807fffff, %edx
	subl	$126, %eax
	orl	$0x3f000000, %edx
	addl	%eax, %ecx
	movl	%edx, VAL(%esp)

	/* Store %ecx in the variable pointed to by the second argument,
	   get the factor from the stack and return.  */
1:	movl	EXPP(%esp), %eax
	flds	VAL(%esp)
	movl	%ecx, (%eax)

	ret
END (__frexpf)
libm_alias_float (__frexp, frexp)

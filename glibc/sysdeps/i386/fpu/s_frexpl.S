/* ix87 specific frexp implementation for long double.
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

#include <libm-alias-ldouble.h>
#include <machine/asm.h>

	.section .rodata

	.align ALIGNARG(4)
	.type two64,@object
two64:	.byte 0, 0, 0, 0, 0, 0, 0xf0, 0x43
	ASM_SIZE_DIRECTIVE(two64)

#ifdef PIC
#define MO(op) op##@GOTOFF(%edx)
#else
#define MO(op) op
#endif

#define PARMS	4		/* no space for saved regs */
#define VAL0	PARMS
#define VAL1	VAL0+4
#define VAL2	VAL1+4
#define EXPP	VAL2+4

	.text
ENTRY (__frexpl)

	movl	VAL0(%esp), %ecx
	movl	VAL2(%esp), %eax
	orl	VAL1(%esp), %ecx
	movl	%eax, %edx
	andl	$0x7fff, %eax
	orl	%eax, %ecx
	jz	1f
	xorl	%ecx, %ecx
	cmpl	$0x7fff, %eax
	je	3f

	cmpl	$0, %eax
	jne	2f

	fldt	VAL0(%esp)
#ifdef	PIC
	LOAD_PIC_REG (dx)
#endif

	fmull	MO(two64)	/* It's not necessary to use a 80bit factor */
	movl	$-64, %ecx
	fstpt	VAL0(%esp)
	fwait
	movl	VAL2(%esp), %eax
	movl	%eax, %edx
	andl	$0x7fff, %eax

2:	andl	$0x8000, %edx
	subl	$16382, %eax
	orl	$0x3ffe, %edx
	addl	%eax, %ecx
	movl	%edx, VAL2(%esp)

	/* Store %ecx in the variable pointed to by the second argument,
	   get the factor from the stack and return.  */
1:	movl	EXPP(%esp), %eax
	fldt	VAL0(%esp)
	movl	%ecx, (%eax)

	ret

	/* Infinity or NaN; ensure signaling NaNs are quieted.  */
3:	movl	EXPP(%esp), %eax
	fldt	VAL0(%esp)
	fadd	%st
	movl	%ecx, (%eax)
	ret
END (__frexpl)
libm_alias_ldouble (__frexp, frexp)

/* Pentium __mpn_add_n -- Add two limb vectors of the same length > 0 and store
   sum in a third limb vector.
   Copyright (C) 1992-2021 Free Software Foundation, Inc.
   This file is part of the GNU MP Library.

   The GNU MP Library is free software; you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 2.1 of the License, or (at your
   option) any later version.

   The GNU MP Library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
   License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with the GNU MP Library; see the file COPYING.LIB.  If not,
   see <https://www.gnu.org/licenses/>.  */

#include "sysdep.h"
#include "asm-syntax.h"

#define PARMS	4+16		/* space for 4 saved regs */
#define RES	PARMS
#define S1	RES+4
#define S2	S1+4
#define SIZE	S2+4

	.text
ENTRY (__mpn_add_n)

	pushl	%edi
	cfi_adjust_cfa_offset (4)
	pushl	%esi
	cfi_adjust_cfa_offset (4)
	pushl	%ebp
	cfi_adjust_cfa_offset (4)
	pushl	%ebx
	cfi_adjust_cfa_offset (4)

	movl	RES(%esp),%edi
	cfi_rel_offset (edi, 12)
	movl	S1(%esp),%esi
	cfi_rel_offset (esi, 8)
	movl	S2(%esp),%ebx
	cfi_rel_offset (ebx, 0)
	movl	SIZE(%esp),%ecx
	movl	(%ebx),%ebp
	cfi_rel_offset (ebp, 4)

	decl	%ecx
	movl	%ecx,%edx
	shrl	$3,%ecx
	andl	$7,%edx
	testl	%ecx,%ecx		/* zero carry flag */
	jz	L(end)
	pushl	%edx
	cfi_adjust_cfa_offset (4)

	ALIGN (3)
L(oop):	movl	28(%edi),%eax		/* fetch destination cache line */
	leal	32(%edi),%edi

L(1):	movl	(%esi),%eax
	movl	4(%esi),%edx
	adcl	%ebp,%eax
	movl	4(%ebx),%ebp
	adcl	%ebp,%edx
	movl	8(%ebx),%ebp
	movl	%eax,-32(%edi)
	movl	%edx,-28(%edi)

L(2):	movl	8(%esi),%eax
	movl	12(%esi),%edx
	adcl	%ebp,%eax
	movl	12(%ebx),%ebp
	adcl	%ebp,%edx
	movl	16(%ebx),%ebp
	movl	%eax,-24(%edi)
	movl	%edx,-20(%edi)

L(3):	movl	16(%esi),%eax
	movl	20(%esi),%edx
	adcl	%ebp,%eax
	movl	20(%ebx),%ebp
	adcl	%ebp,%edx
	movl	24(%ebx),%ebp
	movl	%eax,-16(%edi)
	movl	%edx,-12(%edi)

L(4):	movl	24(%esi),%eax
	movl	28(%esi),%edx
	adcl	%ebp,%eax
	movl	28(%ebx),%ebp
	adcl	%ebp,%edx
	movl	32(%ebx),%ebp
	movl	%eax,-8(%edi)
	movl	%edx,-4(%edi)

	leal	32(%esi),%esi
	leal	32(%ebx),%ebx
	decl	%ecx
	jnz	L(oop)

	popl	%edx
	cfi_adjust_cfa_offset (-4)
L(end):
	decl	%edx			/* test %edx w/o clobbering carry */
	js	L(end2)
	incl	%edx
L(oop2):
	leal	4(%edi),%edi
	movl	(%esi),%eax
	adcl	%ebp,%eax
	movl	4(%ebx),%ebp
	movl	%eax,-4(%edi)
	leal	4(%esi),%esi
	leal	4(%ebx),%ebx
	decl	%edx
	jnz	L(oop2)
L(end2):
	movl	(%esi),%eax
	adcl	%ebp,%eax
	movl	%eax,(%edi)

	sbbl	%eax,%eax
	negl	%eax

	popl	%ebx
	cfi_adjust_cfa_offset (-4)
	cfi_restore (ebx)
	popl	%ebp
	cfi_adjust_cfa_offset (-4)
	cfi_restore (ebp)
	popl	%esi
	cfi_adjust_cfa_offset (-4)
	cfi_restore (esi)
	popl	%edi
	cfi_adjust_cfa_offset (-4)
	cfi_restore (edi)

	ret
END (__mpn_add_n)

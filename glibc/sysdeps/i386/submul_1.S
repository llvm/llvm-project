/* i80386 __mpn_submul_1 -- Multiply a limb vector with a limb and subtract
   the result from a second limb vector.
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

#define PARMS	4+16	/* space for 4 saved regs */
#define RES	PARMS
#define S1	RES+4
#define SIZE	S1+4
#define S2LIMB	SIZE+4

#define res_ptr edi
#define s1_ptr esi
#define sizeP ecx
#define s2_limb ebx

	.text
ENTRY (__mpn_submul_1)

	pushl	%edi
	cfi_adjust_cfa_offset (4)
	pushl	%esi
	cfi_adjust_cfa_offset (4)
	pushl	%ebp
	cfi_adjust_cfa_offset (4)
	pushl	%ebx
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (edi, 12)
	cfi_rel_offset (esi, 8)
	cfi_rel_offset (ebp, 4)
	cfi_rel_offset (ebx, 0)

	movl	RES(%esp), %res_ptr
	movl	S1(%esp), %s1_ptr
	movl	SIZE(%esp), %sizeP
	movl	S2LIMB(%esp), %s2_limb
	leal	(%res_ptr,%sizeP,4), %res_ptr
	leal	(%s1_ptr,%sizeP,4), %s1_ptr
	negl	%sizeP
	xorl	%ebp, %ebp
	ALIGN (3)
L(oop):
	movl	(%s1_ptr,%sizeP,4), %eax
	mull	%s2_limb
	addl	%ebp, %eax
	adcl	$0, %edx
	subl	%eax, (%res_ptr,%sizeP,4)
	adcl	$0, %edx
	movl	%edx, %ebp

	incl	%sizeP
	jnz	L(oop)
	movl	%ebp, %eax

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
END (__mpn_submul_1)

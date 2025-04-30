/* Pentium __mpn_submul_1 -- Multiply a limb vector with a limb and subtract
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
#define size ecx
#define s2_limb ebx

	.text
ENTRY (__mpn_submul_1)

	pushl	%res_ptr
	cfi_adjust_cfa_offset (4)
	pushl	%s1_ptr
	cfi_adjust_cfa_offset (4)
	pushl	%ebp
	cfi_adjust_cfa_offset (4)
	pushl	%s2_limb
	cfi_adjust_cfa_offset (4)

	movl	RES(%esp), %res_ptr
	cfi_rel_offset (res_ptr, 12)
	movl	S1(%esp), %s1_ptr
	cfi_rel_offset (s1_ptr, 8)
	movl	SIZE(%esp), %size
	movl	S2LIMB(%esp), %s2_limb
	cfi_rel_offset (s2_limb, 0)
	leal	(%res_ptr,%size,4), %res_ptr
	leal	(%s1_ptr,%size,4), %s1_ptr
	negl	%size
	xorl	%ebp, %ebp
	cfi_rel_offset (ebp, 4)
	ALIGN (3)

L(oop):	adcl	$0, %ebp
	movl	(%s1_ptr,%size,4), %eax

	mull	%s2_limb

	addl	%ebp, %eax
	movl	(%res_ptr,%size,4), %ebp

	adcl	$0, %edx
	subl	%eax, %ebp

	movl	%ebp, (%res_ptr,%size,4)
	incl	%size

	movl	%edx, %ebp
	jnz	L(oop)

	adcl	$0, %ebp
	movl	%ebp, %eax
	popl	%s2_limb
	cfi_adjust_cfa_offset (-4)
	cfi_restore (s2_limb)
	popl	%ebp
	cfi_adjust_cfa_offset (-4)
	cfi_restore (ebp)
	popl	%s1_ptr
	cfi_adjust_cfa_offset (-4)
	cfi_restore (s1_ptr)
	popl	%res_ptr
	cfi_adjust_cfa_offset (-4)
	cfi_restore (res_ptr)

	ret
#undef size
END (__mpn_submul_1)

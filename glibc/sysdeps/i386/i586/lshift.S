/* Pentium optimized __mpn_lshift --
   Copyright (C) 1992-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#include "sysdep.h"
#include "asm-syntax.h"

#define PARMS	4+16		/* space for 4 saved regs */
#define RES	PARMS
#define S	RES+4
#define SIZE	S+4
#define CNT	SIZE+4

	.text
ENTRY (__mpn_lshift)

	pushl	%edi
	cfi_adjust_cfa_offset (4)
	pushl	%esi
	cfi_adjust_cfa_offset (4)
	pushl	%ebp
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (ebp, 0)
	pushl	%ebx
	cfi_adjust_cfa_offset (4)

	movl	RES(%esp),%edi
	cfi_rel_offset (edi, 12)
	movl	S(%esp),%esi
	cfi_rel_offset (esi, 8)
	movl	SIZE(%esp),%ebx
	cfi_rel_offset (ebx, 0)
	movl	CNT(%esp),%ecx

/* We can use faster code for shift-by-1 under certain conditions.  */
	cmp	$1,%ecx
	jne	L(normal)
	leal	4(%esi),%eax
	cmpl	%edi,%eax
	jnc	L(special)		/* jump if s_ptr + 1 >= res_ptr */
	leal	(%esi,%ebx,4),%eax
	cmpl	%eax,%edi
	jnc	L(special)		/* jump if res_ptr >= s_ptr + size */

L(normal):
	leal	-4(%edi,%ebx,4),%edi
	leal	-4(%esi,%ebx,4),%esi

	movl	(%esi),%edx
	subl	$4,%esi
	xorl	%eax,%eax
	shldl	%cl,%edx,%eax		/* compute carry limb */
	pushl	%eax			/* push carry limb onto stack */
	cfi_adjust_cfa_offset (4)

	decl	%ebx
	pushl	%ebx
	cfi_adjust_cfa_offset (4)
	shrl	$3,%ebx
	jz	L(end)

	movl	(%edi),%eax		/* fetch destination cache line */

	ALIGN	(2)
L(oop):	movl	-28(%edi),%eax		/* fetch destination cache line */
	movl	%edx,%ebp

	movl	(%esi),%eax
	movl	-4(%esi),%edx
	shldl	%cl,%eax,%ebp
	shldl	%cl,%edx,%eax
	movl	%ebp,(%edi)
	movl	%eax,-4(%edi)

	movl	-8(%esi),%ebp
	movl	-12(%esi),%eax
	shldl	%cl,%ebp,%edx
	shldl	%cl,%eax,%ebp
	movl	%edx,-8(%edi)
	movl	%ebp,-12(%edi)

	movl	-16(%esi),%edx
	movl	-20(%esi),%ebp
	shldl	%cl,%edx,%eax
	shldl	%cl,%ebp,%edx
	movl	%eax,-16(%edi)
	movl	%edx,-20(%edi)

	movl	-24(%esi),%eax
	movl	-28(%esi),%edx
	shldl	%cl,%eax,%ebp
	shldl	%cl,%edx,%eax
	movl	%ebp,-24(%edi)
	movl	%eax,-28(%edi)

	subl	$32,%esi
	subl	$32,%edi
	decl	%ebx
	jnz	L(oop)

L(end):	popl	%ebx
	cfi_adjust_cfa_offset (-4)
	andl	$7,%ebx
	jz	L(end2)
L(oop2):
	movl	(%esi),%eax
	shldl	%cl,%eax,%edx
	movl	%edx,(%edi)
	movl	%eax,%edx
	subl	$4,%esi
	subl	$4,%edi
	decl	%ebx
	jnz	L(oop2)

L(end2):
	shll	%cl,%edx		/* compute least significant limb */
	movl	%edx,(%edi)		/* store it */

	popl	%eax			/* pop carry limb */
	cfi_adjust_cfa_offset (-4)

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

/* We loop from least significant end of the arrays, which is only
   permissible if the source and destination don't overlap, since the
   function is documented to work for overlapping source and destination.
*/

	cfi_adjust_cfa_offset (16)
	cfi_rel_offset (edi, 12)
	cfi_rel_offset (esi, 8)
	cfi_rel_offset (ebp, 4)
	cfi_rel_offset (ebx, 0)
L(special):
	movl	(%esi),%edx
	addl	$4,%esi

	decl	%ebx
	pushl	%ebx
	cfi_adjust_cfa_offset (4)
	shrl	$3,%ebx

	addl	%edx,%edx
	incl	%ebx
	decl	%ebx
	jz	L(Lend)

	movl	(%edi),%eax		/* fetch destination cache line */

	ALIGN	(2)
L(Loop):
	movl	28(%edi),%eax		/* fetch destination cache line */
	movl	%edx,%ebp

	movl	(%esi),%eax
	movl	4(%esi),%edx
	adcl	%eax,%eax
	movl	%ebp,(%edi)
	adcl	%edx,%edx
	movl	%eax,4(%edi)

	movl	8(%esi),%ebp
	movl	12(%esi),%eax
	adcl	%ebp,%ebp
	movl	%edx,8(%edi)
	adcl	%eax,%eax
	movl	%ebp,12(%edi)

	movl	16(%esi),%edx
	movl	20(%esi),%ebp
	adcl	%edx,%edx
	movl	%eax,16(%edi)
	adcl	%ebp,%ebp
	movl	%edx,20(%edi)

	movl	24(%esi),%eax
	movl	28(%esi),%edx
	adcl	%eax,%eax
	movl	%ebp,24(%edi)
	adcl	%edx,%edx
	movl	%eax,28(%edi)

	leal	32(%esi),%esi		/* use leal not to clobber carry */
	leal	32(%edi),%edi
	decl	%ebx
	jnz	L(Loop)

L(Lend):
	popl	%ebx
	cfi_adjust_cfa_offset (-4)
	sbbl	%eax,%eax		/* save carry in %eax */
	andl	$7,%ebx
	jz	L(Lend2)
	addl	%eax,%eax		/* restore carry from eax */
L(Loop2):
	movl	%edx,%ebp
	movl	(%esi),%edx
	adcl	%edx,%edx
	movl	%ebp,(%edi)

	leal	4(%esi),%esi		/* use leal not to clobber carry */
	leal	4(%edi),%edi
	decl	%ebx
	jnz	L(Loop2)

	jmp	L(L1)
L(Lend2):
	addl	%eax,%eax		/* restore carry from eax */
L(L1):	movl	%edx,(%edi)		/* store last limb */

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
END (__mpn_lshift)

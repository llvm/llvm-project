/* i80386 __mpn_lshift --
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

#define PARMS	4+12		/* space for 3 saved regs */
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
	pushl	%ebx
	cfi_adjust_cfa_offset (4)

	movl	RES(%esp),%edi
	cfi_rel_offset (edi, 8)
	movl	S(%esp),%esi
	cfi_rel_offset (esi, 4)
	movl	SIZE(%esp),%edx
	movl	CNT(%esp),%ecx
	subl	$4,%esi			/* adjust s_ptr */

	movl	(%esi,%edx,4),%ebx	/* read most significant limb */
	cfi_rel_offset (ebx, 0)
	cfi_remember_state
	xorl	%eax,%eax
	shldl	%cl,%ebx,%eax		/* compute carry limb */
	decl	%edx
	jz	L(end)
	pushl	%eax			/* push carry limb onto stack */
	cfi_adjust_cfa_offset (4)
	testb	$1,%dl
	jnz	L(1)			/* enter loop in the middle */
	movl	%ebx,%eax

	ALIGN (3)
L(oop):	movl	(%esi,%edx,4),%ebx	/* load next lower limb */
	shldl	%cl,%ebx,%eax		/* compute result limb */
	movl	%eax,(%edi,%edx,4)	/* store it */
	decl	%edx
L(1):	movl	(%esi,%edx,4),%eax
	shldl	%cl,%eax,%ebx
	movl	%ebx,(%edi,%edx,4)
	decl	%edx
	jnz	L(oop)

	shll	%cl,%eax		/* compute least significant limb */
	movl	%eax,(%edi)		/* store it */

	popl	%eax			/* pop carry limb */
	cfi_adjust_cfa_offset (-4)

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

	cfi_restore_state
L(end):	shll	%cl,%ebx		/* compute least significant limb */
	movl	%ebx,(%edi)		/* store it */

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
END (__mpn_lshift)

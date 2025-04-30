/* Machine-specific calling sequence for `mcount' profiling function.  SuperH
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   Contributed by NIIBE Yutaka <gniibe@m17n.org>
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

#include <sysdep.h>

	.globl C_SYMBOL_NAME(_mcount)
	.type C_SYMBOL_NAME(_mcount),@function
	cfi_startproc
	.align	5
C_LABEL(_mcount)
	/* Save registers.  */
	mov.l	r4,@-r15
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (r4, 0)
	mov.l	r5,@-r15
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (r5, 0)
	mov.l	r6,@-r15
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (r6, 0)
	mov.l	r7,@-r15
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (r7, 0)
	sts.l	pr,@-r15
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (pr, 0)

	mov.l	@(20,r15),r4
	sts	pr,r5

#ifdef SHARED
	mov.l	0f,r1
	mova	0f,r0
	add	r1,r0
	mov.l	1f,r1
	mov.l	@(r0,r1),r1
#else
	mov.l	1f,r1
#endif
	jsr	@r1
	 nop

	/* Pop the saved registers.  */
	lds.l	@r15+,pr
	cfi_adjust_cfa_offset (-4)
	cfi_restore (pr)
	mov.l	@r15+,r7
	cfi_adjust_cfa_offset (-4)
	cfi_restore (r7)
	mov.l	@r15+,r6
	cfi_adjust_cfa_offset (-4)
	cfi_restore (r6)
	mov.l	@r15+,r5
	cfi_adjust_cfa_offset (-4)
	cfi_restore (r5)
	rts
	 mov.l	@r15+,r4
	/* Omit CFI for restore in delay slot.  */

	.align 2
#ifdef SHARED
0:	.long	_GLOBAL_OFFSET_TABLE_
1:	.long	C_SYMBOL_NAME(__mcount_internal)@GOT
#else
1:	.long	C_SYMBOL_NAME(__mcount_internal)
#endif
	cfi_endproc
ASM_SIZE_DIRECTIVE(C_SYMBOL_NAME(_mcount))

#undef mcount
weak_alias (_mcount, mcount)

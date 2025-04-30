/* Machine-specific calling sequence for `mcount' profiling function.  alpha
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
   Contributed by David Mosberger (davidm@cs.arizona.edu).
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* Assembly stub to invoke _mcount().  Compiler generated code calls
   this stub after executing a function's prologue and without saving any
   registers.  It is therefore necessary to preserve a0..a5 as they may
   contain function arguments.  To work correctly with frame- less
   functions, it is also necessary to preserve ra.  Finally, division
   routines are invoked with a special calling convention and the
   compiler treats those calls as if they were instructions.  In
   particular, it doesn't save any of the temporary registers (caller
   saved registers).  It is therefore necessary to preserve all
   caller-saved registers as well.

   Upon entering _mcount, register $at holds the return address and ra
   holds the return address of the function's caller (selfpc and frompc,
   respectively in gmon.c language...). */

#include <sysdep.h>

	.set	noat
	.set	noreorder

LEAF(_mcount, 0xb0)
	subq	 sp, 0xb0, sp
	.prologue 0
	stq	 a0, 0x00(sp)
	mov	 ra, a0		# a0 = caller-pc
	stq	 a1, 0x08(sp)
	mov	$at, a1		# a1 = self-pc
	stq	$at, 0x10(sp)

	stq	 a2, 0x18(sp)
	stq	 a3, 0x20(sp)
	stq	 a4, 0x28(sp)
	stq	 a5, 0x30(sp)
	stq	 ra, 0x38(sp)
	stq	 gp, 0x40(sp)

	br	gp, 1f
1:	ldgp	gp, 0(gp)

	stq	 t0, 0x48(sp)
	stq	 t1, 0x50(sp)
	stq	 t2, 0x58(sp)
	stq	 t3, 0x60(sp)
	stq	 t4, 0x68(sp)
	stq	 t5, 0x70(sp)
	stq	 t6, 0x78(sp)

	stq	 t7, 0x80(sp)
	stq	 t8, 0x88(sp)
	stq	 t9, 0x90(sp)
	stq	t10, 0x98(sp)
	stq	t11, 0xa0(sp)
	stq	 v0, 0xa8(sp)

	jsr	ra, __mcount

	ldq	 a0, 0x00(sp)
	ldq	 a1, 0x08(sp)
	ldq	$at, 0x10(sp)	# restore self-pc
	ldq	 a2, 0x18(sp)
	ldq	 a3, 0x20(sp)
	ldq	 a4, 0x28(sp)
	ldq	 a5, 0x30(sp)
	ldq	 ra, 0x38(sp)
	ldq	 gp, 0x40(sp)
	mov	$at, pv		# make pv point to return address
	ldq	 t0, 0x48(sp)	# this is important under OSF/1 to
	ldq	 t1, 0x50(sp)	# ensure that the code that we return
	ldq	 t2, 0x58(sp)	# can correctly compute its gp
	ldq	 t3, 0x60(sp)
	ldq	 t4, 0x68(sp)
	ldq	 t5, 0x70(sp)
	ldq	 t6, 0x78(sp)
	ldq	 t7, 0x80(sp)
	ldq	 t8, 0x88(sp)
	ldq	 t9, 0x90(sp)
	ldq	t10, 0x98(sp)
	ldq	t11, 0xa0(sp)
	ldq	 v0, 0xa8(sp)

	addq	sp, 0xb0, sp
	ret	zero,($at),1

	END(_mcount)

weak_alias (_mcount, mcount)

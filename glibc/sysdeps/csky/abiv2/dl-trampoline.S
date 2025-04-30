/* PLT trampolines.  C-SKY ABIV2 version.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

/* This function is not called directly.  It is jumped when attempting to
   use a symbol that has not yet been resolved.

   .plt*:
     subi r0, 32
     stw r2, (r0, 0)
     stw r3, (r0, 4)
     lrw r3, #offset
     ldw r2, (gb, 8)
     jmp r2
 */

#include <sysdep.h>

	.globl _dl_runtime_resolve;
	.type _dl_runtime_resolve,@function;
	.align 4;
_dl_runtime_resolve:
	subi	sp, 20
	stm	a0-a3, (sp)
	stw	lr, (sp, 16)
	/* a0 = id, a1 = offset(do it in plt*).  */
	ldw	a0,  (gb, 4)
	movi	a1, 12
	mult	a1, t1
	bsr	_dl_fixup
	mov	t0, a0
	ldw	lr, (sp, 16)
	ldm	a0-a3, (sp)
	addi	sp, 20
	jmp	t0
	.size _dl_runtime_resolve,.-_dl_runtime_resolve

/* Copyright (C) 1997-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>
#include <jmpbuf-offsets.h>
#include <stap-probe.h>

/* __longjmp(jmpbuf, val) */

ENTRY (__longjmp)
	cfi_def_cfa(x0, 0)
	cfi_offset(x19, JB_X19<<3)
	cfi_offset(x20, JB_X20<<3)
	cfi_offset(x21, JB_X21<<3)
	cfi_offset(x22, JB_X22<<3)
	cfi_offset(x23, JB_X23<<3)
	cfi_offset(x24, JB_X24<<3)
	cfi_offset(x25, JB_X25<<3)
	cfi_offset(x26, JB_X26<<3)
	cfi_offset(x27, JB_X27<<3)
	cfi_offset(x28, JB_X28<<3)
	cfi_offset(x29, JB_X29<<3)
	cfi_offset(x30, JB_LR<<3)

	cfi_offset( d8, JB_D8<<3)
	cfi_offset( d9, JB_D9<<3)
	cfi_offset(d10, JB_D10<<3)
	cfi_offset(d11, JB_D11<<3)
	cfi_offset(d12, JB_D12<<3)
	cfi_offset(d13, JB_D13<<3)
	cfi_offset(d14, JB_D14<<3)
	cfi_offset(d15, JB_D15<<3)

	PTR_ARG (0)

	ldp	x19, x20, [x0, #JB_X19<<3]
	ldp	x21, x22, [x0, #JB_X21<<3]
	ldp	x23, x24, [x0, #JB_X23<<3]
	ldp	x25, x26, [x0, #JB_X25<<3]
	ldp	x27, x28, [x0, #JB_X27<<3]
#ifdef PTR_DEMANGLE
	ldp	x29,  x4, [x0, #JB_X29<<3]
	PTR_DEMANGLE (30, 4, 3, 2)
#else
	ldp	x29, x30, [x0, #JB_X29<<3]
#endif
	/* longjmp probe takes 3 arguments, address of jump buffer as
	   first argument (8@x0), return value as second argument (-4@x1),
	   and target address (8@x30), respectively.  */
	LIBC_PROBE (longjmp, 3, 8@x0, -4@x1, 8@x30)
	ldp	 d8,  d9, [x0, #JB_D8<<3]
	ldp	d10, d11, [x0, #JB_D10<<3]
	ldp	d12, d13, [x0, #JB_D12<<3]
	ldp	d14, d15, [x0, #JB_D14<<3]

        /* Originally this was implemented with a series of
	   .cfi_restore() directives.

           The theory was that cfi_restore should revert to previous
           frame value is the same as the current value.  In practice
           this doesn't work, even after cfi_restore() gdb continues
           to try to recover a previous frame value offset from x0,
           which gets stuffed after a few more instructions.  The
           cfi_same_value() mechanism appears to work fine.  */

	cfi_same_value(x19)
	cfi_same_value(x20)
	cfi_same_value(x21)
	cfi_same_value(x22)
	cfi_same_value(x23)
	cfi_same_value(x24)
	cfi_same_value(x25)
	cfi_same_value(x26)
	cfi_same_value(x27)
	cfi_same_value(x28)
	cfi_same_value(x29)
	cfi_same_value(x30)
	cfi_same_value(d8)
	cfi_same_value(d9)
	cfi_same_value(d10)
	cfi_same_value(d11)
	cfi_same_value(d12)
	cfi_same_value(d13)
	cfi_same_value(d14)
	cfi_same_value(d15)
#ifdef PTR_DEMANGLE
	ldr	x4, [x0, #JB_SP<<3]
	PTR_DEMANGLE (5, 4, 3, 2)
#else
	ldr	x5, [x0, #JB_SP<<3]
#endif
	mov	sp, x5

	/* longjmp_target probe takes 3 arguments, address of jump buffer
	   as first argument (8@x0), return value as second argument (-4@x1),
	   and target address (8@x30), respectively.  */
	LIBC_PROBE (longjmp_target, 3, 8@x0, -4@x1, 8@x30)
	cmp	x1, #0
	mov	x0, #1
	csel	x0, x1, x0, ne
	/* Use br instead of ret because ret is guaranteed to mispredict */
	br	x30
END (__longjmp)

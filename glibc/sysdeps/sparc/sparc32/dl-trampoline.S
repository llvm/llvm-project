/* PLT trampolines.  Sparc 32-bit version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

	.text
	.align	32

	/* %g1:	PLT offset loaded by PLT entry
	 * %g2: callers PC, which is PLT0 + 4, and we store the
	 *      link map at PLT0 + 12, therefore we add 8 to get
	 *      the address of the link map
	 */
	.globl	_dl_runtime_resolve
	.type	_dl_runtime_resolve, @function
_dl_runtime_resolve:
	cfi_startproc

	save	%sp, -104, %sp
	cfi_def_cfa_register(%fp)
	cfi_window_save
	cfi_register (%o7, %i7)

	ld	[%g2 + 8], %o0
	srl	%g1, 10, %o1
	call	_dl_fixup
	 sub	%o1, 4*12, %o1
	jmp	%o0
	 restore

	cfi_endproc

	.size	_dl_runtime_resolve, .-_dl_runtime_resolve

	/* For the profiling cases we pass in our stack frame
	 * as the base of the La_sparc32_regs, so it looks
	 * like:
	 *	%l0			%sp + (0 * 8)
	 *	%l1			%sp + (0 * 8) + 4
	 *	...
	 *	%l6			%sp + (3 * 8)
	 *	%l7			%sp + (3 * 8) + 4
	 *	%i0			%sp + (4 * 8)
	 *	%i1			%sp + (4 * 8) + 4
	 *	...
	 *	%i6			%sp + (7 * 8)
	 *	%i7			%sp + (7 * 8) + 4
	 *	struct_ret_ptr		%sp + (8 * 8)
	 *	framesize		%sp + (9 * 8)
	 */

	.globl	_dl_profile_save_regs
	.type	_dl_profile_save_regs, @function
_dl_profile_save_regs:
	cfi_startproc

	std	%l0, [%sp + ( 0 * 8)]
	std	%l2, [%sp + ( 1 * 8)]
	std	%l4, [%sp + ( 2 * 8)]
	std	%l6, [%sp + ( 3 * 8)]
	std	%i0, [%sp + ( 4 * 8)]
	std	%i2, [%sp + ( 5 * 8)]
	std	%i4, [%sp + ( 6 * 8)]
	std	%i6, [%sp + ( 7 * 8)]
	ld	[%fp + (8 * 8)], %l4
	retl
	 st	%l4, [%sp + (8 * 8)]

	cfi_endproc

	.size	_dl_profile_save_regs, .-_dl_profile_save_regs

	/* If we are going to call pltexit, then we must replicate
	 * the caller's stack frame.
	 * %o0: PLT resolved function address
	 */
	.globl	_dl_profile_invoke
	.type	_dl_profile_invoke, @function
_dl_profile_invoke:
	cfi_startproc

	add	%l0, 7, %l0
	andn	%l0, 7, %l0
	add	%l0, 2 * 8, %g1

	sub	%sp, %g1, %sp
	srl	%l0, 3, %l7
	mov	%o0, %l1
	mov	%i0, %o0
	mov	%i1, %o1
	mov	%i2, %o2
	mov	%i3, %o3
	mov	%i4, %o4
	mov	%i5, %o5
	cmp	%l0, 0
	mov	%fp, %l2
	be	2f
	 add	%sp, (11 * 8), %l3
1:	ldd	[%l2], %g2
	add	%l2, 0x8, %l2
	subcc	%l7, 1, %l7
	std	%g2, [%l3]
	bne	1b
	 add	%l3, 0x8, %l3

2:	jmpl	%l1, %o7
	 nop

	std	%o0, [%sp + ( 9 * 8)]
	std	%f0, [%sp + (10 * 8)]

	mov	%l5, %o0
	mov	%l6, %o1
	add	%sp, (11 * 8), %o2
	call	_dl_call_pltexit
	 add	%sp, ( 9 * 8), %o3

	ldd	[%sp + ( 9 * 8)], %i0
	ldd	[%sp + (10 * 8)], %f0

	jmpl	%i7 + 8, %g0
	 restore

	cfi_endproc

	.size	_dl_profile_invoke, .-_dl_profile_invoke

	/* %g1:	PLT offset loaded by PLT entry
	 * %g2: callers PC, which is PLT0 + 4, and we store the
	 *      link map at PLT0 + 12, therefore we add 8 to get
	 *      the address of the link map
	 */
	.align	32
	.globl	_dl_runtime_profile
	.type	_dl_runtime_profile, @function
_dl_runtime_profile:
	cfi_startproc

	save	%sp, -104, %sp
	cfi_def_cfa_register(%fp)
	cfi_window_save
	cfi_register(%o7, %i7)

	ld	[%g2 + 8], %o0
	srl	%g1, 10, %o1
	mov	%i7, %o2
	sub	%o1, 4*12, %o1

	mov	%o0, %l5
	mov	%o1, %l6

	call	_dl_profile_save_regs
	 nop

	mov	%sp, %o3
	call	_dl_profile_fixup
	 add	%sp, (9 * 8), %o4

	ld	[%sp + (9 * 8)], %l0
	cmp	%l0, 0
	bl	1f
	 nop

	call	_dl_profile_invoke
	 nop

1:	jmp	%o0
	 restore

	cfi_endproc

	.size	_dl_runtime_profile, .-_dl_runtime_profile

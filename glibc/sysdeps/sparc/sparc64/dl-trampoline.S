/* PLT trampolines.  Sparc 64-bit version.
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
	 * %g4: callers PC, which is PLT0 + 24, therefore we
	 *      add (32 + 8) to get the address of PLT2 which
	 *      is where the magic cookie is stored
	 */
	.globl	_dl_runtime_resolve_0
	.type	_dl_runtime_resolve_0, @function
_dl_runtime_resolve_0:
	cfi_startproc

	save	%sp, -192, %sp
	cfi_def_cfa_register(%fp)
	cfi_window_save
	cfi_register(%o7, %i7)

	sethi   %hi(1047552), %l2
	ldx	[%g4 + 32 + 8], %o0
	sub     %g1, %g4, %l0
	xor     %l2, -1016, %l2
	sethi   %hi(5120), %l3	/* 160 * 32 */
	add     %l0, %l2, %l0
	sethi   %hi(32768), %l4
	udivx   %l0, %l3, %l3
	sllx    %l3, 2, %l1
	add     %l1, %l3, %l1
	sllx    %l1, 10, %l2
	sub	%l4, 4, %l4
	sllx    %l1, 5, %l1
	sub     %l0, %l2, %l0
	udivx   %l0, 24, %l0
	add     %l0, %l4, %l0
	add     %l1, %l0, %l1
	add     %l1, %l1, %l0
	add     %l0, %l1, %l0
	call	_dl_fixup
	 sllx    %l0, 3, %o1
	jmp	%o0
	 restore

	cfi_endproc

	.size	_dl_runtime_resolve_0, .-_dl_runtime_resolve_0

	/* %g1:	PLT offset loaded by PLT entry
	 * %g4: callers PC, which is PLT1 + 24, therefore we
	 *      add 8 to get the address of PLT2 which
	 *      is where the magic cookie is stored
	 */
	.globl	_dl_runtime_resolve_1
	.type	_dl_runtime_resolve_1, @function
_dl_runtime_resolve_1:
	cfi_startproc

	save	%sp, -192, %sp
	cfi_def_cfa_register(%fp)
	cfi_window_save
	cfi_register(%o7, %i7)

	srlx	%g1, 12, %o1
	ldx	[%g4 + 8], %o0
	add	%o1, %o1, %o3
	sub	%o1, 96, %o1
	call	_dl_fixup
	 add	%o1, %o3, %o1
	jmp	%o0
	 restore

	cfi_endproc

	.size	_dl_runtime_resolve_1, .-_dl_runtime_resolve_1

	/* For the profiling cases we pass in our stack frame
	 * as the base of the La_sparc64_regs, so it looks
	 * like:
	 *	%l0			%sp
	 *	...
	 *	%l7			%sp + (7 * 8)
	 *	%i0			%sp + (8 * 8)
	 *	...
	 *	%i7			%sp + (15 * 8)
	 *	%f0			%sp + (16 * 8)
	 *	%f16			%sp + (31 * 8)
	 *	framesize		%sp + (32 * 8)
	 */

	.globl	_dl_profile_save_regs
	.type	_dl_profile_save_regs, @function
_dl_profile_save_regs:
	cfi_startproc

	stx	%l0, [%sp + STACK_BIAS + ( 0 * 8)]
	stx	%l1, [%sp + STACK_BIAS + ( 1 * 8)]
	stx	%l2, [%sp + STACK_BIAS + ( 2 * 8)]
	stx	%l3, [%sp + STACK_BIAS + ( 3 * 8)]
	stx	%l4, [%sp + STACK_BIAS + ( 4 * 8)]
	stx	%l5, [%sp + STACK_BIAS + ( 5 * 8)]
	stx	%l6, [%sp + STACK_BIAS + ( 6 * 8)]
	stx	%l7, [%sp + STACK_BIAS + ( 7 * 8)]
	stx	%i0, [%sp + STACK_BIAS + ( 8 * 8)]
	stx	%i1, [%sp + STACK_BIAS + ( 9 * 8)]
	stx	%i2, [%sp + STACK_BIAS + (10 * 8)]
	stx	%i3, [%sp + STACK_BIAS + (11 * 8)]
	stx	%i4, [%sp + STACK_BIAS + (12 * 8)]
	stx	%i5, [%sp + STACK_BIAS + (13 * 8)]
	stx	%i6, [%sp + STACK_BIAS + (14 * 8)]
	stx	%i7, [%sp + STACK_BIAS + (15 * 8)]
	std	%f0, [%sp + STACK_BIAS + (16 * 8)]
	std	%f2, [%sp + STACK_BIAS + (17 * 8)]
	std	%f4, [%sp + STACK_BIAS + (18 * 8)]
	std	%f6, [%sp + STACK_BIAS + (19 * 8)]
	std	%f8, [%sp + STACK_BIAS + (20 * 8)]
	std	%f10, [%sp + STACK_BIAS + (21 * 8)]
	std	%f12, [%sp + STACK_BIAS + (22 * 8)]
	std	%f14, [%sp + STACK_BIAS + (23 * 8)]
	std	%f16, [%sp + STACK_BIAS + (24 * 8)]
	std	%f18, [%sp + STACK_BIAS + (25 * 8)]
	std	%f20, [%sp + STACK_BIAS + (26 * 8)]
	std	%f22, [%sp + STACK_BIAS + (27 * 8)]
	std	%f24, [%sp + STACK_BIAS + (28 * 8)]
	std	%f26, [%sp + STACK_BIAS + (29 * 8)]
	std	%f28, [%sp + STACK_BIAS + (30 * 8)]
	retl
	 std	%f30, [%sp + STACK_BIAS + (31 * 8)]

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
	add	%l0, (8 * 8), %g1

	sub	%sp, %g1, %sp
	srlx	%l0, 3, %l7
	mov	%o0, %l1
	mov	%i0, %o0
	mov	%i1, %o1
	mov	%i2, %o2
	mov	%i3, %o3
	mov	%i4, %o4
	mov	%i5, %o5
	add	%fp, STACK_BIAS, %l2
	brz	%l0, 2f
	 add	%sp, STACK_BIAS, %l3
1:	ldx	[%l2], %l4
	add	%l2, 0x8, %l2
	subcc	%l7, 1, %l7
	stx	%l4, [%l3]
	bne,pt	%xcc, 1b
	 add	%l3, 0x8, %l3

2:	jmpl	%l1, %o7
	 nop

	stx	%o0, [%sp + STACK_BIAS + (16 * 8)]
	stx	%o1, [%sp + STACK_BIAS + (17 * 8)]
	stx	%o2, [%sp + STACK_BIAS + (18 * 8)]
	stx	%o3, [%sp + STACK_BIAS + (19 * 8)]
	std	%f0, [%sp + STACK_BIAS + (20 * 8)]
	std	%f2, [%sp + STACK_BIAS + (21 * 8)]
	std	%f4, [%sp + STACK_BIAS + (22 * 8)]
	std	%f8, [%sp + STACK_BIAS + (23 * 8)]

	mov	%l5, %o0
	mov	%l6, %o1
	add	%sp, STACK_BIAS + (24 * 8), %o2
	call	_dl_call_pltexit
	 add	%sp, STACK_BIAS + (16 * 8), %o3

	ldx	[%sp + STACK_BIAS + (16 * 8)], %i0
	ldx	[%sp + STACK_BIAS + (17 * 8)], %i1
	ldx	[%sp + STACK_BIAS + (18 * 8)], %i2
	ldx	[%sp + STACK_BIAS + (19 * 8)], %i3
	ldd	[%sp + STACK_BIAS + (20 * 8)], %f0
	ldd	[%sp + STACK_BIAS + (21 * 8)], %f2
	ldd	[%sp + STACK_BIAS + (22 * 8)], %f4
	ldd	[%sp + STACK_BIAS + (23 * 8)], %f8

	jmpl	%i7 + 8, %g0
	 restore

	cfi_endproc

	.size	_dl_profile_invoke, .-_dl_profile_invoke

	/* %g1:	PLT offset loaded by PLT entry
	 * %g4: callers PC, which is PLT0 + 24, therefore we
	 *      add (32 + 8) to get the address of PLT2 which
	 *      is where the magic cookie is stored
	 */
	.align	32
	.globl	_dl_runtime_profile_0
	.type	_dl_runtime_profile_0, @function
_dl_runtime_profile_0:
	cfi_startproc

	save	%sp, -336, %sp
	cfi_def_cfa_register(%fp)
	cfi_window_save
	cfi_register(%o7, %i7)

	sethi   %hi(1047552), %l2
	ldx	[%g4 + 32 + 8], %o0
	sub     %g1, %g4, %l0
	xor     %l2, -1016, %l2
	sethi   %hi(5120), %l3	/* 160 * 32 */
	add     %l0, %l2, %l0
	sethi   %hi(32768), %l4
	udivx   %l0, %l3, %l3
	sllx    %l3, 2, %l1
	add     %l1, %l3, %l1
	sllx    %l1, 10, %l2
	sub	%l4, 4, %l4
	sllx    %l1, 5, %l1
	sub     %l0, %l2, %l0
	udivx   %l0, 24, %l0
	add     %l0, %l4, %l0
	add     %l1, %l0, %l1
	add     %l1, %l1, %l0
	add     %l0, %l1, %l0

	mov	%i7, %o2
	sllx    %l0, 3, %o1

	mov	%o0, %l5
	mov	%o1, %l6

	call	_dl_profile_save_regs
	 nop

	add	%sp, STACK_BIAS, %o3
	call	_dl_profile_fixup
	 add	%sp, (STACK_BIAS + (32 * 8)), %o4

	ldx	[%sp + STACK_BIAS + (32 * 8)], %l0
	brlz,pt %l0, 1f
	 nop

	call	_dl_profile_invoke
	 nop

1:	jmp	%o0
	 restore

	cfi_endproc

	.size	_dl_runtime_profile_0, .-_dl_runtime_profile_0

	/* %g1:	PLT offset loaded by PLT entry
	 * %g4: callers PC, which is PLT1 + 24, therefore we
	 *      add 8 to get the address of PLT2 which
	 *      is where the magic cookie is stored
	 */
	.globl	_dl_runtime_profile_1
	.type	_dl_runtime_profile_1, @function
_dl_runtime_profile_1:
	cfi_startproc

	save	%sp, -336, %sp
	cfi_def_cfa_register(%fp)
	cfi_window_save
	cfi_register(%o7, %i7)

	srlx	%g1, 12, %o1
	ldx	[%g4 + 8], %o0
	add	%o1, %o1, %o3
	sub	%o1, 96, %o1
	mov	%i7, %o2
	add	%o1, %o3, %o1

	mov	%o0, %l5
	mov	%o1, %l6

	call	_dl_profile_save_regs
	 nop

	add	%sp, STACK_BIAS, %o3
	call	_dl_profile_fixup
	 add	%sp, (STACK_BIAS + (32 * 8)), %o4

	ldx	[%sp + STACK_BIAS + (32 * 8)], %l0
	brlz,pt %l0, 1f
	 nop

	call	_dl_profile_invoke
	 nop

1:	jmp	%o0
	 restore

	cfi_endproc

	.size	_dl_runtime_resolve_1, .-_dl_runtime_resolve_1

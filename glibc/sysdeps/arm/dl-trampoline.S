/* PLT trampolines.  ARM version.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* ??? Needs more rearrangement for the LDM to handle thumb mode.  */
#define NO_THUMB
#include <sysdep.h>
#include <libc-symbols.h>

	.text
	.globl _dl_runtime_resolve
	.type _dl_runtime_resolve, #function
	CFI_SECTIONS
	cfi_startproc
	.align 2
_dl_runtime_resolve:
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (lr, 0)

	@ we get called with
	@ 	stack[0] contains the return address from this call
	@	ip contains &GOT[n+3] (pointer to function)
	@	lr points to &GOT[2]

	@ Save arguments.  We save r4 to realign the stack.
	push	{r0-r4}
	cfi_adjust_cfa_offset (20)
	cfi_rel_offset (r0, 0)
	cfi_rel_offset (r1, 4)
	cfi_rel_offset (r2, 8)
	cfi_rel_offset (r3, 12)

	@ get pointer to linker struct
	ldr	r0, [lr, #-4]

	@ prepare to call _dl_fixup()
	@ change &GOT[n+3] into 8*n        NOTE: reloc are 8 bytes each
	sub	r1, ip, lr
	sub	r1, r1, #4
	add	r1, r1, r1

	@ call fixup routine
	bl	_dl_fixup

	@ save the return
	mov	ip, r0

	@ get arguments and return address back.  We restore r4
	@ only to realign the stack.
	pop	{r0-r4,lr}
	cfi_adjust_cfa_offset (-24)

	@ jump to the newly found address
	BX(ip)

	cfi_endproc
	.size _dl_runtime_resolve, .-_dl_runtime_resolve

#ifndef PROF
	.globl _dl_runtime_profile
	.type _dl_runtime_profile, #function
	CFI_SECTIONS
	cfi_startproc
	.align 2
_dl_runtime_profile:
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (lr, 0)

	@ we get called with
	@ 	stack[0] contains the return address from this call
	@	ip contains &GOT[n+3] (pointer to function)
	@	lr points to &GOT[2]

	@ Stack layout:
	@ 212 - saved lr
	@ 208 - framesize returned from pltenter
	@ 16 - La_arm_regs
	@ 8 - Saved two arguments to _dl_profile_fixup
	@ 4 - Saved result of _dl_profile_fixup
	@ 0 - outgoing argument to _dl_profile_fixup
	@ For now, we only save the general purpose registers.

	sub	sp, sp, #196
	cfi_adjust_cfa_offset (196)
	stmia	sp, {r0-r3}
	cfi_rel_offset (r0, 0)
	cfi_rel_offset (r1, 4)
	cfi_rel_offset (r2, 8)
	cfi_rel_offset (r3, 12)

	sub	sp, sp, #16
	cfi_adjust_cfa_offset (16)

	@ Save sp and lr.
	add	r0, sp, #216
	str	r0, [sp, #32]
	ldr	r2, [sp, #212]
	str	r2, [sp, #36]

	@ get pointer to linker struct
	ldr	r0, [lr, #-4]

	@ prepare to call _dl_profile_fixup()
	@ change &GOT[n+3] into 8*n        NOTE: reloc are 8 bytes each
	sub	r1, ip, lr
	sub	r1, r1, #4
	add	r1, r1, r1

	@ Save these two arguments for pltexit.
	add	r3, sp, #8
	stmia	r3!, {r0,r1}

	@ Set up extra args for _dl_profile_fixup.
	@ r2 and r3 are already loaded.
	add	ip, sp, #208
	str	ip, [sp, #0]

	@ call profiling fixup routine
	bl	_dl_profile_fixup

	@ The address to call is now in r0.

	@ Check whether we're wrapping this function.
	ldr	ip, [sp, #208]
	cmp	ip, #0
	bge	1f
	cfi_remember_state

	@ save the return
	mov	ip, r0

	@ get arguments and return address back
	add	sp, sp, #16
	cfi_adjust_cfa_offset (-16)
	ldmia	sp, {r0-r3,sp,lr}
	cfi_adjust_cfa_offset (-200)

	@ jump to the newly found address
	BX(ip)

	cfi_restore_state
1:
	@ The new frame size is in ip.

	@ New stack layout:
	@ 268 - saved r7
	@ 264 - saved result of _dl_profile_fixup
	@ 72 - La_arm_regs
	@ 64 - Saved two arguments to _dl_profile_fixup
	@ 0 - La_arm_retval
	@ For now, we only save the general purpose registers.

	@ Build the new frame.
	str	r7, [sp, #212]
	cfi_rel_offset (r7, 212)
	sub	r7, sp, #56
	cfi_def_cfa_register (r7)
	cfi_adjust_cfa_offset (56)
	sub	sp, sp, ip
	bic	sp, sp, #7

	@ Save the _dl_profile_fixup result around the call to memcpy.
	str	r0, [r7, #264]

	@ Copy the stack arguments.
	mov	r0, sp
	add	r1, r7, #272
	mov	r2, ip
	bl	memcpy

	@ Call the function.
	add	ip, r7, #72
	ldmia	ip, {r0-r3}
	ldr	ip, [r7, #264]
	BLX(ip)
	stmia	r7, {r0-r3}

	@ Call pltexit.
	add	ip, r7, #64
	ldmia	ip, {r0,r1}
	add	r2, r7, #72
	add	r3, r7, #0
	bl	_dl_call_pltexit

	@ Return to caller.
	ldmia	r7, {r0-r3}
	mov	sp, r7
	cfi_def_cfa_register (sp)
	ldr	r7, [sp, #268]
	ldr	lr, [sp, #92]
	add	sp, sp, #272
	cfi_adjust_cfa_offset (-272)
	BX(lr)

	cfi_endproc
	.size _dl_runtime_profile, .-_dl_runtime_profile
#endif
	.previous

/* PLT trampolines.  Nios II version.
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

#include <sysdep.h>
#include <libc-symbols.h>

	.text
	.globl _dl_runtime_resolve
	cfi_startproc
_dl_runtime_resolve:
/* The runtime resolver receives the original function arguments in r4
   through r7, the shared library identifier from GOT[1]? in r14, and the
   relocation index times four in r15. It updates the corresponding PLT GOT
   entry so that the PLT entry will transfer control directly to the target
   in the future, and then transfers control to the target. */
	/* Save arguments and return address.  */
	subi sp, sp, 28
	cfi_adjust_cfa_offset (28)
	stw r22, 24(sp)
	cfi_rel_offset (r22, 24)
	stw r8, 20(sp)  /* save r8, because this might be a call to mcount */
	cfi_rel_offset (r8, 20)
	stw r7, 16(sp)
	cfi_rel_offset (r7, 16)
	stw r6, 12(sp)
	cfi_rel_offset (r6, 12)
	stw r5, 8(sp)
	cfi_rel_offset (r5, 8)
	stw r4, 4(sp)
	cfi_rel_offset (r4, 4)
	stw ra, 0(sp)
	cfi_rel_offset (ra, 0)

	/* Get pointer to linker struct.  */
	mov r4, r14

	/* Get the relocation offset.  We're given a multiple of 4 and
	   need a multiple of 12, so multiply by 3. */
	slli r5, r15, 1
	add r5, r5, r15

	/* Call the fixup routine.  */
	nextpc r22
1:	movhi r2, %hiadj(_gp_got - 1b)
	addi r2, r2, %lo(_gp_got - 1b)
	add r22, r22, r2
	ldw r2, %call(_dl_fixup)(r22)
	callr r2

	/* Restore the arguments and return address.  */
	ldw ra, 0(sp)
	ldw r4, 4(sp)
	ldw r5, 8(sp)
	ldw r6, 12(sp)
	ldw r7, 16(sp)
	ldw r8, 20(sp)
	ldw r22, 24(sp)
	addi sp, sp, 28
	cfi_adjust_cfa_offset (-28)

	/* Jump to the newly found address.  */
	jmp r2

	cfi_endproc

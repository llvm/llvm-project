/* PLT trampolines.  ARC version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>
#include <sys/syscall.h>

/* resolver has atypical calling ABI (r11 and r12)
        PLTn which lands us here, sets up
	r11 = Module info (tpnt pointer as expected by resolver)
	r12 = PC of the PLTn itself - needed by resolver to find
	      corresponding .rela.plt entry.  */

ENTRY (_dl_runtime_resolve)

	/* save args to func being resolved before entering resolver.  */
	push_s	r0
	push_s	r1
	push_s	r2
	push_s	r3
	st.a	r4, [sp, -4]
	st.a	r5, [sp, -4]
	st.a	r6, [sp, -4]
	st.a	r7, [sp, -4]
	st.a	r8, [sp, -4]
	st.a	r9, [sp, -4]
	cfi_adjust_cfa_offset (40)
	push_s	blink
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (blink, 0)

	mov_s 	r1, r12
	bl.d  	_dl_fixup
	mov   	r0, r11

	/* restore regs back.  */
	ld.ab	blink,[sp, 4]
	cfi_adjust_cfa_offset (-4)
	cfi_restore (blink)
	ld.ab	r9, [sp, 4]
	ld.ab	r8, [sp, 4]
	ld.ab	r7, [sp, 4]
	ld.ab	r6, [sp, 4]
	ld.ab	r5, [sp, 4]
	ld.ab	r4, [sp, 4]
	pop_s   r3
	pop_s   r2
	pop_s   r1
	cfi_adjust_cfa_offset (-36)

	j_s.d   [r0]    /* r0 has resolved function addr.  */
	pop_s   r0      /* restore first arg to resolved call.  */
	cfi_adjust_cfa_offset (-4)
	cfi_restore (r0)
END (_dl_runtime_resolve)

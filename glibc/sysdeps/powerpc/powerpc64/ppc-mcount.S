/* PowerPC64-specific implementation of profiling support.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
/* We don't need to save the parameter-passing registers as gcc takes
   care of that for us.  Thus this function looks fairly normal.
   In fact, the generic code would work for us.  */

ENTRY(_mcount)
	mflr		 r4
	ld		 r11, 0(r1)
	std		 r4, FRAME_LR_SAVE(r1)
	stdu		 r1,-FRAME_MIN_SIZE(r1)
	cfi_adjust_cfa_offset (FRAME_MIN_SIZE)
	cfi_offset (lr, FRAME_LR_SAVE)
	ld		 r3, FRAME_LR_SAVE(r11)
	bl		 JUMPTARGET (NOTOC (__mcount_internal))
#ifndef SHARED
	nop
#endif
	ld		 r0, FRAME_MIN_SIZE+FRAME_LR_SAVE(r1)
	mtlr		 r0
	addi		 r1,r1,FRAME_MIN_SIZE
	blr
END(_mcount)

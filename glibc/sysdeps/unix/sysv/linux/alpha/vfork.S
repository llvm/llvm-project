/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
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
#include <tcb-offsets.h>

ENTRY(__libc_vfork)
	PSEUDO_PROLOGUE

	/* Load the thread pointer value in A1 across the vfork.  */
	rduniq
	mov	v0, a1

	lda	v0, SYS_ify(vfork)
	call_pal PAL_callsys

	/* Normal error check and return.  */
	bne	a3, SYSCALL_ERROR_LABEL
	ret

PSEUDO_END (__libc_vfork)

#if IS_IN (libc)
weak_alias (__libc_vfork, vfork)
strong_alias (__libc_vfork, __vfork)
libc_hidden_def (__vfork)
#endif

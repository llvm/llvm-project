/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <errno.h>

#if IS_IN (rtld)
# include <dl-sysdep.h>			/* Defines RTLD_PRIVATE_ERRNO.  */
#endif

#include <tls.h>

#undef syscall_error
__syscall_error:
#if defined (EWOULDBLOCK_sys) && EWOULDBLOCK_sys != EAGAIN
	/* We translate the system's EWOULDBLOCK error into EAGAIN.
	   The GNU C library always defines EWOULDBLOCK==EAGAIN.
	   EWOULDBLOCK_sys is the original number.  */
	cmp	r0, $EWOULDBLOCK_sys /* Is it the old EWOULDBLOCK?  */
	it	eq
	moveq	r0, $EAGAIN	/* Yes; translate it to EAGAIN.  */
#endif

#if !IS_IN (rtld)
	mov	r1, r0
	GET_TLS (r2)
	ldr	r2, 1f
#ifdef __thumb__
2:	add	r2, r2, pc
	ldr	r2, [r2]
#else
2:	ldr	r2, [pc, r2]
#endif
	str	r1, [r0, r2]
	mvn	r0, #0
	DO_RET(lr)

1:	.word errno(gottpoff) + (. - 2b - PC_OFS)
#elif RTLD_PRIVATE_ERRNO
	LDST_PCREL(str, r0, r1, C_SYMBOL_NAME(rtld_errno))
	mvn	r0, #0
	DO_RET(r14)
#else
#error "Unsupported non-TLS case"
#endif

#undef	__syscall_error
END (__syscall_error)

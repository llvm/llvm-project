/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
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
#include <errno.h>
#include <tls.h>

#if IS_IN (rtld)
# include <dl-sysdep.h>		/* Defines RTLD_PRIVATE_ERRNO.  */
#endif

.globl C_SYMBOL_NAME(errno)
.globl syscall_error

__syscall_error:
#if defined (EWOULDBLOCK_sys) && EWOULDBLOCK_sys != EAGAIN
	/* We translate the system's EWOULDBLOCK error into EAGAIN.
	   The GNU C library always defines EWOULDBLOCK==EAGAIN.
	   EWOULDBLOCK_sys is the original number.  */
	cmp $EWOULDBLOCK_sys, %RAX_LP /* Is it the old EWOULDBLOCK?  */
	jne notb		/* Branch if not.  */
	movl $EAGAIN, %eax	/* Yes; translate it to EAGAIN.  */
notb:
#endif
#ifdef PIC
	movq C_SYMBOL_NAME(errno@GOTTPOFF)(%rip), %rcx
	movl %eax, %fs:0(%rcx)
#else
	movl %eax, %fs:C_SYMBOL_NAME(errno@TPOFF)
#endif
	or $-1, %RAX_LP
	ret

#undef	__syscall_error
END (__syscall_error)

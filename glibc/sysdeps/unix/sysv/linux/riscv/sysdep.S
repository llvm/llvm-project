/* syscall error handlers
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
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

#if IS_IN (libc)
# define errno __libc_errno
#endif

ENTRY (__syscall_error)
	mv t0, ra
	/* Fall through to __syscall_set_errno.  */
END (__syscall_error)

/* Non-standard calling convention: argument in a0, return address in t0,
   and clobber only t1.  */
ENTRY (__syscall_set_errno)
	/* We got here because a0 < 0, but only codes in the range [-4095, -1]
	  represent errors.  Otherwise, just return the result normally.  */
	li t1, -4096
	bleu a0, t1, 1f
	neg a0, a0
#if RTLD_PRIVATE_ERRNO
	sw a0, rtld_errno, t1
#elif defined(__PIC__)
	la.tls.ie t1, errno
	add t1, t1, tp
	sw a0, 0(t1)
#else
	lui t1, %tprel_hi(errno)
	add t1, t1, tp, %tprel_add(errno)
	sw a0, %tprel_lo(errno)(t1)
#endif
	li a0, -1
1:	jr t0
END (__syscall_set_errno)

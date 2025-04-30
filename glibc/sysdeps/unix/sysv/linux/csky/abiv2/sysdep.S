/* syscall error handlers.  C-SKY ABIV2 version.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

/* The syscall stubs jump here when they detect an error.
   The code for Linux is almost identical to the canonical Unix
   code, except that the error number in R0 is negated.  */

#undef CALL_MCOUNT
#define CALL_MCOUNT /* Don't insert the profiling call, it clobbers R0.  */

	.text
ENTRY (__syscall_error)
	movi	a1, 0
	rsub	a0, a0, a1

#if !IS_IN (rtld)
	mov	a1, a0
	mov	a0, tls

	grs	t1, .Lgetpc1
.Lgetpc1:
	lrw	t0, errno@gottpoff
	add	t1, t1, t0
	ldw	t1, (t1)
	add	t1, a0
	stw	a1, (t1)
	bmaski	a0, 0
	rts
#elif RTLD_PRIVATE_ERRNO /* !IS_IN (rtld) */
# ifdef  __PIC__
	grs	t1, .Lgetpc2
.Lgetpc2:
	lrw	t0, .Lgetpc2@GOTPC
	addu	t1, t1, t0
	lrw	t0, rtld_errno@PLT
	ldr.w	t0, (t1, t0 << 0)
# else
	lrw	t0, rtld_errno
# endif /* __PIC__ */
	stw	a0, (t0)
	bmaski	a0, 0
	rts
#else
# error "Unsupported non-TLS case"
#endif /* RTLD_PRIVATE_ERRNO */

#undef __syscall_error
END (__syscall_error)

/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Brendan Kehoe (brendan@zen.org).

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
#include <sys/asm.h>

	.set	nomips16

#ifdef _LIBC_REENTRANT

LOCALSZ= 3
FRAMESZ= (((NARGSAVE+LOCALSZ)*SZREG)+ALSZ)&ALMASK
RAOFF= FRAMESZ-(1*SZREG)
GPOFF= FRAMESZ-(2*SZREG)
V0OFF= FRAMESZ-(3*SZREG)

ENTRY(__syscall_error)
#ifdef __PIC__
	.set noat
	SETUP_GPX (AT)
	.set at
#endif
	PTR_SUBU sp, FRAMESZ
	.set noat
	SETUP_GPX64(GPOFF,AT)
	.set at
#ifdef __PIC__
	SAVE_GP(GPOFF)
#endif
	REG_S	v0, V0OFF(sp)
	REG_S	ra, RAOFF(sp)

#if defined (EWOULDBLOCK_sys) && EWOULDBLOCK_sys != EAGAIN
	/* We translate the system's EWOULDBLOCK error into EAGAIN.
	   The GNU C library always defines EWOULDBLOCK==EAGAIN.
	   EWOULDBLOCK_sys is the original number.  */
	bne	v0, EWOULDBLOCK_sys, L(skip)
	nop
	li	v0, EAGAIN
L(skip):
#endif
	/* Find our per-thread errno address  */
	jal	__errno_location

	/* Store the error value.  */
	REG_L	t0, V0OFF(sp)
	sw	t0, 0(v0)

	/* And just kick back a -1.  */
	REG_L	ra, RAOFF(sp)
	RESTORE_GP64
	PTR_ADDU sp, FRAMESZ
	li	v0, -1
	j	ra
	END(__syscall_error)

#else /* _LIBC_REENTRANT */


ENTRY(__syscall_error)
#ifdef __PIC__
	SETUP_GPX (AT)
#endif
	SETUP_GPX64 (t9, AT)

#if defined (EWOULDBLOCK_sys) && EWOULDBLOCK_sys != EAGAIN
	/* We translate the system's EWOULDBLOCK error into EAGAIN.
	   The GNU C library always defines EWOULDBLOCK==EAGAIN.
	   EWOULDBLOCK_sys is the original number.  */
	bne v0, EWOULDBLOCK_sys, L(skip)
	li v0, EAGAIN
L(skip):
#endif
	/* Store it in errno... */
	sw v0, errno

	/* And just kick back a -1.  */
	li v0, -1

	RESTORE_GP64
	j ra
	END(__syscall_error)
#endif  /* _LIBC_REENTRANT  */

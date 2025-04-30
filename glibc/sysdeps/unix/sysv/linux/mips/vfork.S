/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

/* vfork() is just a special case of clone().  */

#include <sys/asm.h>
#include <sysdep.h>
#include <sgidefs.h>
#include <tls.h>


/* int vfork() */

	.text
	.set		nomips16
LOCALSZ= 1
FRAMESZ= (((NARGSAVE+LOCALSZ)*SZREG)+ALSZ)&ALMASK
GPOFF= FRAMESZ-(1*SZREG)
NESTED(__libc_vfork,FRAMESZ,sp)
#ifdef __PIC__
	SETUP_GP
#endif
	PTR_SUBU sp, FRAMESZ
	cfi_adjust_cfa_offset (FRAMESZ)
	SETUP_GP64_REG (a5, __libc_vfork)
#ifdef __PIC__
	SAVE_GP (GPOFF)
#endif
#ifdef PROF
# if (_MIPS_SIM != _ABIO32)
	PTR_S		a5, GPOFF(sp)
# endif
	.set		noat
	move		$1, ra
# if (_MIPS_SIM == _ABIO32)
	subu		sp,sp,8
# endif
	jal		_mcount
	.set		at
# if (_MIPS_SIM != _ABIO32)
	PTR_L		a5, GPOFF(sp)
# endif
#endif

	PTR_ADDU	sp, FRAMESZ
	cfi_adjust_cfa_offset (-FRAMESZ)

	li		a0, 0x4112	/* CLONE_VM | CLONE_VFORK | SIGCHLD */
	move		a1, sp

	/* Do the system call */
	li		v0,__NR_clone
	syscall

	cfi_remember_state
	bnez		a3,L(error)

	/* Successful return from the parent or child.  */
	RESTORE_GP64_REG
	ret

	/* Something bad happened -- no child created.  */
L(error):
	cfi_restore_state
#ifdef __PIC__
	PTR_LA		t9, __syscall_error
	RESTORE_GP64_REG
	jr		t9
#else
	RESTORE_GP64_REG
	j		__syscall_error
#endif
	END(__libc_vfork)

#if IS_IN (libc)
weak_alias (__libc_vfork, vfork)
strong_alias (__libc_vfork, __vfork)
libc_hidden_def (__vfork)
#endif

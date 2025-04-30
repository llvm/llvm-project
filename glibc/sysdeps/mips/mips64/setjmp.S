/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#include <sgidefs.h>
#include <sysdep.h>
#include <sys/asm.h>

/* The function __sigsetjmp_aux saves all the registers, but it can't
   reliably access the stack or frame pointers, so we pass them in as
   extra arguments.  */
#ifdef __PIC__
	.option pic2
#endif
ENTRY (__sigsetjmp)
#ifdef __PIC__
	SETUP_GP
#endif
	SETUP_GP64_REG (v0, C_SYMBOL_NAME (__sigsetjmp))
	move a2, sp
	move a3, fp
	PTR_LA t9, __sigsetjmp_aux
#if _MIPS_SIM == _ABIO32
	nop
#endif
	RESTORE_GP64_REG
#if _MIPS_SIM != _ABIO32
	move a4, gp
#endif
	jr t9
END (__sigsetjmp)
hidden_def (__sigsetjmp)

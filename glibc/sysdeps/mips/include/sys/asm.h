/* Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#ifndef _SYS_ASM_H
#include_next <sys/asm.h>

# ifndef _ISOMAC

# undef __mips_cfi_startproc
# define __mips_cfi_startproc cfi_startproc
# undef __mips_cfi_endproc
# define __mips_cfi_endproc cfi_endproc

# if _MIPS_SIM == _ABIO32
#  define SETUP_GP64_REG_CFI(a)
#  define SETUP_GP64_REG(a, b)
#  define SETUP_GP64_STACK_CFI(a)
#  define SETUP_GP64_STACK(a, b)
#  define RESTORE_GP64_REG
#  define RESTORE_GP64_STACK
# else
#  define SETUP_GP64_REG_CFI(gpsavereg)		\
	cfi_register (gp, gpsavereg)
#  define SETUP_GP64_REG(gpsavereg, proc)	\
	SETUP_GP64 (gpsavereg, proc);		\
	SETUP_GP64_REG_CFI (gpsavereg)
#  define SETUP_GP64_STACK_CFI(gpoffset)	\
	cfi_rel_offset (gp, gpoffset)
#  define SETUP_GP64_STACK(gpoffset, proc)	\
	SETUP_GP64 (gpoffset, proc);		\
	SETUP_GP64_STACK_CFI (gpoffset)
#  define RESTORE_GP64_REG			\
	RESTORE_GP64;				\
	cfi_restore (gp)
#  define RESTORE_GP64_STACK			\
	RESTORE_GP64;				\
	cfi_restore (gp)
# endif

# endif /* _ISOMAC */
#endif /* sys/asm.h */

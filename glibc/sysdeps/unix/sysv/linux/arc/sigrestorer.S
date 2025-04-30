/* Default sigreturn stub for ARC Linux.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <sys/syscall.h>
#include <sysdep.h>
#include <tcb-offsets.h>

/* Note the NOP has to be outside body.  */
	nop
ENTRY (__default_rt_sa_restorer)
	mov r8, __NR_rt_sigreturn
	ARC_TRAP_INSN
	j_s     [blink]
PSEUDO_END_NOERRNO (__default_rt_sa_restorer)

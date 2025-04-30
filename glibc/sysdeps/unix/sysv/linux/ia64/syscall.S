/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jes Sorensen <Jes.Sorensen@cern.ch>.

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

ENTRY(syscall)
	/* We are called like so:
	   {out0,out1,...,out6} registers -> {NR, arg1, ..., arg6}
	   Shift the register window so that {out1...out6} are available
	   in {out0...out5} like the kernel syscall handler expects.  */
	alloc r2=ar.pfs,1,0,8,0
	mov r15=r32		/* syscall number */
	break __IA64_BREAK_SYSCALL
	;;
	cmp.ne p6,p0=-1,r10	/* r10 = -1 on error */
(p6)	ret
	br.cond.spnt.few __syscall_error
PSEUDO_END(syscall)

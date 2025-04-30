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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

ENTRY (syscall)
	mr   r0,r3
	mr   r3,r4
	mr   r4,r5
	mr   r5,r6
	mr   r6,r7
	mr   r7,r8
	mr   r8,r9
#if defined(USE_PPC_SCV) && !IS_IN(rtld) && (defined(__PPC64__) || defined(__powerpc64__))
	CHECK_SCV_SUPPORT r9 0f
	DO_CALL_SCV
	RET_SCV
	b 1f
#endif
0:	DO_CALL_SC
	RET_SC
1:	TAIL_CALL_SYSCALL_ERROR
PSEUDO_END (syscall)

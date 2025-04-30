/* Sigreturn stub function used on sa_restore field.
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

#include <sysdep.h>

/* These functions must not change the register window or the stack
   pointer [1].

   [1] https://lkml.org/lkml/2016/5/27/465  */

ENTRY (__rt_sigreturn_stub)
	mov	__NR_rt_sigreturn, %g1
	ta	0x10
END (__rt_sigreturn_stub)

ENTRY (__sigreturn_stub)
	mov	__NR_sigreturn, %g1
	ta	0x10
END (__sigreturn_stub)

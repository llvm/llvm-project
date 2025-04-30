/* Checking mempcpy for x86-64.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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
#include "asm-syntax.h"

#ifndef SHARED
	/* For libc.so this is defined in memcpy.S.
	   For libc.a, this is a separate source to avoid
	   mempcpy bringing in __chk_fail and all routines
	   it calls.  */
        .text
ENTRY (__mempcpy_chk)
	cmpq	%rdx, %rcx
	jb	__chk_fail
	jmp	mempcpy
END (__mempcpy_chk)
#endif

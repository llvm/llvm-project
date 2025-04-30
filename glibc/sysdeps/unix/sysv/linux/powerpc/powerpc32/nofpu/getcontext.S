/* Save current context.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
#include <rtld-global-offsets.h>
#include <shlib-compat.h>

#define __ASSEMBLY__
#include <asm/ptrace.h>
#include "ucontext_i.h"

#define __CONTEXT_FUNC_NAME __getcontext
#undef __CONTEXT_ENABLE_FPRS
#undef __CONTEXT_ENABLE_VRS

#include "getcontext-common.S"

versioned_symbol (libc, __getcontext, getcontext, GLIBC_2_3_4)

#if SHLIB_COMPAT (libc, GLIBC_2_3_3, GLIBC_2_3_4)

/* For the nofpu case the old/new versions are the same function.  */
strong_alias (__getcontext, __novec_getcontext)

compat_symbol (libc, __novec_getcontext, getcontext, GLIBC_2_3_3)

#endif

#if SHLIB_COMPAT (libc, GLIBC_2_1, GLIBC_2_3_3)

#define _ERRNO_H	1
#include <bits/errno.h>

	compat_text_section
ENTRY (__getcontext_stub)
	li	r3,ENOSYS
	b	__syscall_error@local
END (__getcontext_stub)
	.previous

compat_symbol (libc, __getcontext_stub, getcontext, GLIBC_2_1)

#endif

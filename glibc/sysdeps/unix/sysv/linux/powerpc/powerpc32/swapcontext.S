/* Save current context and jump to a new context.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>
#include <rtld-global-offsets.h>
#include <shlib-compat.h>

#define __ASSEMBLY__
#include <asm/ptrace.h>
#include "ucontext_i.h"

#define __CONTEXT_FUNC_NAME __swapcontext
#define __CONTEXT_ENABLE_FPRS 1
#define __CONTEXT_ENABLE_VRS 1

/* Size of ucontext in GLIBC_2.3.4 and later.  */
#define _UC_SIZE_2_3_4	1184

	.section ".text";
ENTRY (__swapcontext)
	li	r5,_UC_SIZE_2_3_4;
	DO_CALL (SYS_ify (swapcontext));
	bso-	cr0,1f
/* the kernel does not set the return code for the success case */
	li	r3,0
	blr
1:
	b	__syscall_error@local
END(__swapcontext)

versioned_symbol (libc, __swapcontext, swapcontext, GLIBC_2_3_4)

#if SHLIB_COMPAT (libc, GLIBC_2_3_3, GLIBC_2_3_4)

	compat_text_section

# undef __CONTEXT_FUNC_NAME
# define __CONTEXT_FUNC_NAME __novec_swapcontext
# undef __CONTEXT_ENABLE_VRS

# include "swapcontext-common.S"

	.previous

compat_symbol (libc, __novec_swapcontext, swapcontext, GLIBC_2_3_3)

#endif

#if SHLIB_COMPAT (libc, GLIBC_2_1, GLIBC_2_3_3)

# define _ERRNO_H	1
# include <bits/errno.h>

	compat_text_section
ENTRY (__swapcontext_stub)
	li	r3,ENOSYS
	b	__syscall_error@local
END (__swapcontext_stub)
	.previous

compat_symbol (libc, __swapcontext_stub, swapcontext, GLIBC_2_1)

#endif

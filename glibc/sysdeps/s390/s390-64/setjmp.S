/* setjmp for 64 bit S/390, ELF version.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).
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
#define _ASM
#define _SETJMP_H
#include <bits/setjmp.h>
#include <shlib-compat.h>
#include <stap-probe.h>

#if !IS_IN (rtld) && defined SHARED \
  && SHLIB_COMPAT (libc, GLIBC_2_19, GLIBC_2_20)
# define NEED_COMPAT_SYMBOLS 1
/* We need a unique name in case of symbol versioning.  */
# define setjmp __v1setjmp
# define _setjmp __v1_setjmp
# define __sigsetjmp __v1__sigsetjmp
#else
# define NEED_COMPAT_SYMBOLS 0
#endif

	/* We include the BSD entry points here as well.  */
ENTRY (setjmp)
	lghi   %r3,1                /* Second argument of one.  */
	j      .Linternal_sigsetjmp /* Branch relativ to __sigsetjmp.  */
END (setjmp)

	/* Binary compatibility entry point.  */
ENTRY(_setjmp)
	slgr   %r3,%r3              /* Second argument of zero.  */
	j      .Linternal_sigsetjmp /* Branch relativ to __sigsetjmp.  */
END (_setjmp)
#if NEED_COMPAT_SYMBOLS
strong_alias (_setjmp, __GI__setjmp)
#else
libc_hidden_def (_setjmp)
#endif

ENTRY(__setjmp)
	slgr   %r3,%r3              /* Second argument of zero.  */
	j      .Linternal_sigsetjmp /* Branch relativ to __sigsetjmp.  */
END (__setjmp)

ENTRY(__sigsetjmp)
.Linternal_sigsetjmp:
	/* setjmp probe expects sig/setjmp first argument (8@%r2), second
	   argument (-8@%r3) and target address (8@%r14).  */
	LIBC_PROBE (setjmp, 3, 8@%r2, -4@%r3, 8@%r14)
#ifdef PTR_MANGLE
	stmg   %r6,%r13,0(%r2)      /* Store registers in jmp_buf.  */
	lgr    %r4,%r14
	lgr    %r5,%r15
	PTR_MANGLE (%r4, %r1)
	PTR_MANGLE2 (%r5, %r1)
	stmg   %r4,%r5,64(%r2)
#else
	stmg   %r6,%r15,0(%r2)      /* Store registers in jmp_buf.  */
#endif
	std    %f8,80(%r2)
	std    %f9,88(%r2)
	std    %f10,96(%r2)
	std    %f11,104(%r2)
	std    %f12,112(%r2)
	std    %f13,120(%r2)
	std    %f14,128(%r2)
	std    %f15,136(%r2)
#if IS_IN (rtld)
	/* In ld.so we never save the signal mask.  */
	lghi   %r2,0
	br     %r14
#elif defined PIC
	jg     __sigjmp_save@PLT    /* Branch to PLT of __sigsetjmp.  */
#else
	jg     __sigjmp_save
#endif
END (__sigsetjmp)
#if NEED_COMPAT_SYMBOLS
strong_alias (__sigsetjmp, __GI___sigsetjmp)
#else
libc_hidden_def (__sigsetjmp)
#endif

#if NEED_COMPAT_SYMBOLS
/* In glibc release 2.19 new versions of setjmp-functions were introduced,
   but were reverted before 2.20. Thus both versions are the same function.  */

# undef setjmp
# undef _setjmp
# undef __sigsetjmp

strong_alias (__v1setjmp, __v2setjmp);
versioned_symbol (libc, __v1setjmp, setjmp, GLIBC_2_0);
compat_symbol (libc, __v2setjmp, setjmp, GLIBC_2_19);

strong_alias (__v1_setjmp, __v2_setjmp);
versioned_symbol (libc, __v1_setjmp, _setjmp, GLIBC_2_0);
compat_symbol (libc, __v2_setjmp, _setjmp, GLIBC_2_19);

strong_alias (__v1__sigsetjmp, __v2__sigsetjmp);
versioned_symbol (libc, __v1__sigsetjmp, __sigsetjmp, GLIBC_2_0);
compat_symbol (libc, __v2__sigsetjmp, __sigsetjmp, GLIBC_2_19);
#endif /* NEED_COMPAT_SYMBOLS */

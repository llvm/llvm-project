/* setjmp for s390, ELF version.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
	lhi    %r3,1                /* second argument of one */
	j      .Linternal_sigsetjmp /* branch relativ to __sigsetjmp */
END (setjmp)

	/* Binary compatibility entry point.  */
ENTRY(_setjmp)
	lhi    %r3,0                /* second argument of zero */
	j      .Linternal_sigsetjmp /* branch relativ to __sigsetjmp */
END (_setjmp)
#if NEED_COMPAT_SYMBOLS
strong_alias (_setjmp, __GI__setjmp)
#else
libc_hidden_def (_setjmp)
#endif

ENTRY(__setjmp)
	lhi    %r3,0                /* second argument of zero */
	j      .Linternal_sigsetjmp /* branch relativ to __sigsetjmp */
END (__setjmp)

ENTRY(__sigsetjmp)
.Linternal_sigsetjmp:
	/* setjmp probe expects sig/setjmp first argument (4@%r2), second
	   argument (-4@%r3) and target address (4@%r14).  */
	LIBC_PROBE (setjmp, 3, 4@%r2, -4@%r3, 4@%r14)
#ifdef PTR_MANGLE
	stm    %r6,%r13,0(%r2)      /* store registers in jmp_buf */
	lr     %r4,%r14
	lr     %r5,%r15
	PTR_MANGLE (%r4, %r1)
	PTR_MANGLE2 (%r5, %r1)
	stm    %r4,%r5,32(%r2)
#else
	stm    %r6,%r15,0(%r2)      /* store registers in jmp_buf */
#endif
	std    %f4,40(%r2)
	std    %f6,48(%r2)
#if IS_IN (rtld)
	/* In ld.so we never save the signal mask.  */
	lhi    %r2,0
	br     %r14
#elif defined PIC
	/* We cannot use the PLT, because it requires that %r12 be set, but
	   we can't save and restore our caller's value.  Instead, we do an
	   indirect jump through the GOT. */
	basr   %r1,0
.L0:    al     %r1,.L1 - .L0(0,%r1) /* get address of global offset table */
				    /* get address of __sigjmp_save from got */
	l      %r1,__sigjmp_save@GOT(%r1)
	br     %r1
.L1:    .long  _GLOBAL_OFFSET_TABLE_ - .L0
#else
	basr   %r1,0
.L0:    l      %r1,.L1-.L0(0,%r1)   /* load address of __sigjmp_save */
	br     %r1                  /* tail-call __sigjmp_save */
.L1:    .long  __sigjmp_save
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

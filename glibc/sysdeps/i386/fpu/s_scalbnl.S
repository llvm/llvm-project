/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Changes for long double by Ulrich Drepper <drepper@cygnus.com>
 * Public domain.
 */

#include <machine/asm.h>

RCSID("$NetBSD: $")

ENTRY(__scalbnl)
	fildl	16(%esp)
	fldt	4(%esp)
	fscale
	fstp	%st(1)
	ret
END (__scalbnl)
strong_alias (__scalbnl, __scalblnl)

#include <shlib-compat.h>
#if SHLIB_COMPAT (libc, GLIBC_2_1, GLIBC_2_20)
compat_symbol (libc, __scalbnl, scalblnl, GLIBC_2_1);
#endif

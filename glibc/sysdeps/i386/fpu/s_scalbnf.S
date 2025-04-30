/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <i386-math-asm.h>

RCSID("$NetBSD: s_scalbnf.S,v 1.3 1995/05/09 00:19:59 jtc Exp $")

ENTRY(__scalbnf)
	fildl	8(%esp)
	flds	4(%esp)
	fscale
	fstp	%st(1)
	FLT_NARROW_EVAL
	ret
END (__scalbnf)
strong_alias (__scalbnf, __scalblnf)

#include <shlib-compat.h>
#if SHLIB_COMPAT (libc, GLIBC_2_1, GLIBC_2_20)
compat_symbol (libc, __scalbnf, scalblnf, GLIBC_2_1);
#endif

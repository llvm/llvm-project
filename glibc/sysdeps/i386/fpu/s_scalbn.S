/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <i386-math-asm.h>

RCSID("$NetBSD: s_scalbn.S,v 1.4 1995/05/09 00:19:06 jtc Exp $")

ENTRY(__scalbn)
	fildl	12(%esp)
	fldl	4(%esp)
	fscale
	fstp	%st(1)
	DBL_NARROW_EVAL
	ret
END (__scalbn)
strong_alias (__scalbn, __scalbln)

#include <shlib-compat.h>
#if SHLIB_COMPAT (libc, GLIBC_2_1, GLIBC_2_20)
compat_symbol (libc, __scalbn, scalbln, GLIBC_2_1);
#endif

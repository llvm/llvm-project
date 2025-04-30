/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>

RCSID("$NetBSD: s_significandf.S,v 1.3 1995/05/09 00:24:07 jtc Exp $")

ENTRY(__significandf)
	flds	4(%esp)
	fxtract
	fstp	%st(1)
	ret
END (__significandf)
weak_alias (__significandf, significandf)

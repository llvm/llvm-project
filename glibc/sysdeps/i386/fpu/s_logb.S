/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <libm-alias-double.h>

RCSID("$NetBSD: s_logb.S,v 1.4 1995/05/09 00:14:30 jtc Exp $")

ENTRY(__logb)
	fldl	4(%esp)
	fxtract
	fstp	%st
	ret
END (__logb)
libm_alias_double (__logb, logb)

/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <libm-alias-float.h>

RCSID("$NetBSD: s_rintf.S,v 1.3 1995/05/09 00:17:22 jtc Exp $")

ENTRY(__rintf)
	flds	4(%esp)
	frndint
	ret
END (__rintf)
libm_alias_float (__rint, rint)

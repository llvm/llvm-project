/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <i386-math-asm.h>
#include <libm-alias-double.h>

RCSID("$NetBSD: s_atan.S,v 1.4 1995/05/08 23:50:41 jtc Exp $")

DEFINE_DBL_MIN

#ifdef PIC
# define MO(op) op##@GOTOFF(%ecx)
#else
# define MO(op) op
#endif

	.text
ENTRY(__atan)
#ifdef  PIC
	LOAD_PIC_REG (cx)
#endif
	fldl	4(%esp)
	fld1
	fpatan
	DBL_CHECK_FORCE_UFLOW
	ret
END (__atan)
libm_alias_double (__atan, atan)

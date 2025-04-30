/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <i386-math-asm.h>
#include <libm-alias-finite.h>

RCSID("$NetBSD: e_atan2.S,v 1.4 1995/05/08 23:46:28 jtc Exp $")

DEFINE_DBL_MIN

#ifdef PIC
# define MO(op) op##@GOTOFF(%ecx)
#else
# define MO(op) op
#endif

	.text
ENTRY(__ieee754_atan2)
#ifdef  PIC
	LOAD_PIC_REG (cx)
#endif
	fldl	 4(%esp)
	fldl	12(%esp)
	fpatan
	DBL_CHECK_FORCE_UFLOW_NARROW
	ret
END (__ieee754_atan2)
libm_alias_finite (__ieee754_atan2, __atan2)

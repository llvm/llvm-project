/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <i386-math-asm.h>
#include <libm-alias-finite.h>

RCSID("$NetBSD: e_atan2f.S,v 1.1 1995/05/08 23:35:10 jtc Exp $")

DEFINE_FLT_MIN

#ifdef PIC
# define MO(op) op##@GOTOFF(%ecx)
#else
# define MO(op) op
#endif

	.text
ENTRY(__ieee754_atan2f)
#ifdef  PIC
	LOAD_PIC_REG (cx)
#endif
	flds	4(%esp)
	flds	8(%esp)
	fpatan
	FLT_CHECK_FORCE_UFLOW_NARROW
	ret
END (__ieee754_atan2f)
libm_alias_finite (__ieee754_atan2f, __atan2f)

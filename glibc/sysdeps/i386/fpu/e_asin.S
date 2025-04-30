/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <i386-math-asm.h>
#include <libm-alias-finite.h>

RCSID("$NetBSD: e_asin.S,v 1.4 1995/05/08 23:45:40 jtc Exp $")

DEFINE_DBL_MIN

#ifdef PIC
# define MO(op) op##@GOTOFF(%ecx)
#else
# define MO(op) op
#endif

	.text

/* asin = atan (x / sqrt((1-x) (1+x))) */
ENTRY(__ieee754_asin)
#ifdef  PIC
	LOAD_PIC_REG (cx)
#endif
	fldl	4(%esp)			/* x */
	fld	%st
	fld1				/* 1 : x : x */
	fsubp				/* 1 - x : x */
	fld1				/* 1 : 1 - x : x */
	fadd	%st(2)			/* 1 + x : 1 - x : x */
	fmulp				/* 1 - x^2 */
	fsqrt				/* sqrt (1 - x^2) */
	fpatan
	DBL_CHECK_FORCE_UFLOW
	ret
END (__ieee754_asin)
libm_alias_finite (__ieee754_asin, __asin)

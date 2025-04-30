/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 * Adapted for float type by Ulrich Drepper <drepper@cygnus.com>.
 */

#include <machine/asm.h>
#include <i386-math-asm.h>
#include <libm-alias-finite.h>

RCSID("$NetBSD: $")

	.section .rodata.cst4,"aM",@progbits,4

DEFINE_FLT_MIN

#ifdef PIC
# define MO(op) op##@GOTOFF(%ecx)
#else
# define MO(op) op
#endif

	.text

/* asin = atan (x / sqrt(1 - x^2)) */
ENTRY(__ieee754_asinf)
#ifdef  PIC
	LOAD_PIC_REG (cx)
#endif
	flds	4(%esp)			/* x */
	fld	%st
	fmul	%st(0)			/* x^2 */
	fld1
	fsubp				/* 1 - x^2 */
	fsqrt				/* sqrt (1 - x^2) */
	fpatan
	FLT_CHECK_FORCE_UFLOW
	ret
END (__ieee754_asinf)
libm_alias_finite (__ieee754_asinf, __asinf)

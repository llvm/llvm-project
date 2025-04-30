/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <i386-math-asm.h>
#include <libm-alias-float.h>

RCSID("$NetBSD: s_atanf.S,v 1.3 1995/05/08 23:51:33 jtc Exp $")

DEFINE_FLT_MIN

#ifdef PIC
# define MO(op) op##@GOTOFF(%ecx)
#else
# define MO(op) op
#endif

	.text
ENTRY(__atanf)
#ifdef  PIC
	LOAD_PIC_REG (cx)
#endif
	flds	4(%esp)
	fld1
	fpatan
	FLT_CHECK_FORCE_UFLOW
	ret
END (__atanf)
libm_alias_float (__atan, atan)

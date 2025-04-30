/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <i386-math-asm.h>

RCSID("$NetBSD: s_log1p.S,v 1.7 1995/05/09 00:10:58 jtc Exp $")

	.section .rodata

	.align ALIGNARG(4)
	/* The fyl2xp1 can only be used for values in
		-1 + sqrt(2) / 2 <= x <= 1 - sqrt(2) / 2
	   0.29 is a safe value.
	*/
limit:	.double 0.29
one:	.double 1.0

DEFINE_DBL_MIN

#ifdef PIC
# define MO(op) op##@GOTOFF(%edx)
#else
# define MO(op) op
#endif

/*
 * Use the fyl2xp1 function when the argument is in the range -0.29 to 0.29,
 * otherwise fyl2x with the needed extra computation.
 */
	.text
ENTRY(__log1p)
	fldln2

	fldl	4(%esp)

#ifdef	PIC
	LOAD_PIC_REG (dx)
#endif

	fxam
	fnstsw
	fld	%st
	sahf
	jc	3f		// in case x is NaN or ±Inf
4:	fabs
	fcompl	MO(limit)
	fnstsw
	sahf
	jc	2f

	faddl	MO(one)
	fyl2x
	ret

2:	fyl2xp1
	DBL_CHECK_FORCE_UFLOW_NONNAN
	ret

3:	jp	4b		// in case x is ±Inf
	fstp	%st(1)
	fstp	%st(1)
	ret

END (__log1p)

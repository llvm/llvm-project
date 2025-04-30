/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Adapted for use as log2 by Ulrich Drepper <drepper@cygnus.com>.
 * Public domain.
 *
 * Changed to use fyl2xp1 for values near 1, <drepper@cygnus.com>.
 */

#include <machine/asm.h>
#include <libm-alias-finite.h>

	.section .rodata.cst8,"aM",@progbits,8

	.p2align 3
	.type one,@object
one:	.double 1.0
	ASM_SIZE_DIRECTIVE(one)
	/* It is not important that this constant is precise.  It is only
	   a value which is known to be on the safe side for using the
	   fyl2xp1 instruction.  */
	.type limit,@object
limit:	.double 0.29
	ASM_SIZE_DIRECTIVE(limit)


#ifdef PIC
# define MO(op) op##@GOTOFF(%edx)
#else
# define MO(op) op
#endif

	.text
ENTRY(__ieee754_log2l)
#ifdef PIC
	LOAD_PIC_REG (dx)
#endif
	fldl	MO(one)
	fldt	4(%esp)		// x : 1
	fxam
	fnstsw
	fld	%st		// x : x : 1
	sahf
	jc	3f		// in case x is NaN or ±Inf
4:	fsub	%st(2), %st	// x-1 : x : 1
	fld	%st		// x-1 : x-1 : x : 1
	fabs			// |x-1| : x-1 : x : 1
	fcompl	MO(limit)	// x-1 : x : 1
	fnstsw			// x-1 : x : 1
	andb	$0x45, %ah
	jz	2f
	fxam
	fnstsw
	andb	$0x45, %ah
	cmpb	$0x40, %ah
	jne	5f
	fabs			// log2(1) is +0 in all rounding modes.
5:	fstp	%st(1)		// x-1 : 1
	fyl2xp1			// log(x)
	ret

2:	fstp	%st(0)		// x : 1
	fyl2x			// log(x)
	ret

3:	jp	4b		// in case x is ±Inf
	fstp	%st(1)
	fstp	%st(1)
	fadd	%st(0)
	ret
END (__ieee754_log2l)
libm_alias_finite (__ieee754_log2l, __log2l)

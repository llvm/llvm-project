/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <libm-alias-double.h>

RCSID("$NetBSD: s_floor.S,v 1.4 1995/05/09 00:01:59 jtc Exp $")

ENTRY(__floor)
	fldl	4(%esp)
	subl	$32,%esp
	cfi_adjust_cfa_offset (32)

	fnstenv	4(%esp)			/* store fpu environment */

	/* We use here %edx although only the low 1 bits are defined.
	   But none of the operations should care and they are faster
	   than the 16 bit operations.  */
	movl	$0x400,%edx		/* round towards -oo */
	orl	4(%esp),%edx
	andl	$0xf7ff,%edx
	movl	%edx,(%esp)
	fldcw	(%esp)			/* load modified control word */

	frndint				/* round */

	fldenv	4(%esp)			/* restore original environment */

	addl	$32,%esp
	cfi_adjust_cfa_offset (-32)
	ret
END (__floor)
libm_alias_double (__floor, floor)

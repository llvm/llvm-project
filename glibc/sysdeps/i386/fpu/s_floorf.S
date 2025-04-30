/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <libm-alias-float.h>

RCSID("$NetBSD: s_floorf.S,v 1.3 1995/05/09 00:04:32 jtc Exp $")

ENTRY(__floorf)
	flds	4(%esp)
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
END (__floorf)
libm_alias_float (__floor, floor)

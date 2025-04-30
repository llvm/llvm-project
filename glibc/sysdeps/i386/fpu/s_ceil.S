/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <libm-alias-double.h>

RCSID("$NetBSD: s_ceil.S,v 1.4 1995/05/08 23:52:13 jtc Exp $")

ENTRY(__ceil)
	fldl	4(%esp)
	subl	$32,%esp
	cfi_adjust_cfa_offset (32)

	fnstenv	4(%esp)			/* store fpu environment */

	/* We use here %edx although only the low 1 bits are defined.
	   But none of the operations should care and they are faster
	   than the 16 bit operations.  */
	movl	$0x0800,%edx		/* round towards +oo */
	orl	4(%esp),%edx
	andl	$0xfbff,%edx
	movl	%edx,(%esp)
	fldcw	(%esp)			/* load modified control word */

	frndint				/* round */

	fldenv	4(%esp)			/* restore original environment */

	addl	$32,%esp
	cfi_adjust_cfa_offset (-32)
	ret
END (__ceil)
libm_alias_double (__ceil, ceil)

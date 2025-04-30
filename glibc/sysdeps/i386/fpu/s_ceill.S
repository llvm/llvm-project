/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Changes for long double by Ulrich Drepper <drepper@cygnus.com>
 * Public domain.
 */

#include <libm-alias-ldouble.h>
#include <machine/asm.h>

RCSID("$NetBSD: $")

ENTRY(__ceill)
	fldt	4(%esp)
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

	/* Preserve "invalid" exceptions from sNaN input.  */
	fnstsw
	andl	$0x1, %eax
	orl	%eax, 8(%esp)

	fldenv	4(%esp)			/* restore original environment */

	addl	$32,%esp
	cfi_adjust_cfa_offset (-32)
	ret
END (__ceill)
libm_alias_ldouble (__ceil, ceil)

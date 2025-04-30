/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Changes for long double by Ulrich Drepper <drepper@cygnus.com>
 * Changes for x86-64 by Andreas Jaeger <aj@suse.de>
 * Public domain.
 */

#include <libm-alias-ldouble.h>
#include <machine/asm.h>


ENTRY(__ceill)
	fldt	8(%rsp)

	fnstenv	-28(%rsp)		/* store fpu environment */

	/* We use here %edx although only the low 1 bits are defined.
	   But none of the operations should care and they are faster
	   than the 16 bit operations.  */
	movl	$0x0800,%edx		/* round towards +oo */
	orl	-28(%rsp),%edx
	andl	$0xfbff,%edx
	movl	%edx,-32(%rsp)
	fldcw	-32(%rsp)		/* load modified control word */

	frndint				/* round */

	/* Preserve "invalid" exceptions from sNaN input.  */
	fnstsw
	andl	$0x1, %eax
	orl	%eax, -24(%rsp)

	fldenv	-28(%rsp)		/* restore original environment */

	ret
END (__ceill)
libm_alias_ldouble (__ceil, ceil)

/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */
/* Adapted for use as nearbyint by Ulrich Drepper <drepper@cygnus.com>.  */

#include <libm-alias-ldouble.h>
#include <machine/asm.h>

ENTRY(__nearbyintl)
	fldt	8(%rsp)
	fnstenv	-28(%rsp)
	frndint
	fnstsw
	andl	$0x1, %eax
	orl	%eax, -24(%rsp)
	fldenv	-28(%rsp)
	ret
END (__nearbyintl)
libm_alias_ldouble (__nearbyint, nearbyint)

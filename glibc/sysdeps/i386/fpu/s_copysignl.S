/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Changes for long double by Ulrich Drepper <drepper@cygnus.com>
 * Public domain.
 */

#include <libm-alias-ldouble.h>
#include <machine/asm.h>

RCSID("$NetBSD: $")

ENTRY(__copysignl)
	movl	24(%esp),%edx
	movl	12(%esp),%eax
	andl	$0x8000,%edx
	andl	$0x7fff,%eax
	orl	%edx,%eax
	movl	%eax,12(%esp)
	fldt	4(%esp)
	ret
END (__copysignl)
libm_alias_ldouble (__copysign, copysign)

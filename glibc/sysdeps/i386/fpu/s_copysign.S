/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <libm-alias-double.h>

RCSID("$NetBSD: s_copysign.S,v 1.4 1995/05/08 23:53:02 jtc Exp $")

ENTRY(__copysign)
	movl	16(%esp),%edx
	movl	8(%esp),%eax
	andl	$0x80000000,%edx
	andl	$0x7fffffff,%eax
	orl	%edx,%eax
	movl	%eax,8(%esp)
	fldl	4(%esp)
	ret
END (__copysign)
libm_alias_double (__copysign, copysign)

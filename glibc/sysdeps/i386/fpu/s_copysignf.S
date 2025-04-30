/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <libm-alias-float.h>

RCSID("$NetBSD: s_copysignf.S,v 1.3 1995/05/08 23:53:25 jtc Exp $")

ENTRY(__copysignf)
	movl	8(%esp),%edx
	movl	4(%esp),%eax
	andl	$0x80000000,%edx
	andl	$0x7fffffff,%eax
	orl	%edx,%eax
	movl	%eax,4(%esp)
	flds	4(%esp)
	ret
END (__copysignf)
libm_alias_float (__copysign, copysign)

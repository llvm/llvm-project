/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 *
 * Adapted for `long double' by Ulrich Drepper <drepper@cygnus.com>.
 * Adapted for x86-64 by Andreas Jaeger <aj@suse.de>.
 */

#include <machine/asm.h>
#include <libm-alias-finite.h>

RCSID("$NetBSD: $")

ENTRY(__ieee754_fmodl)
	fldt	24(%rsp)
	fldt	8(%rsp)
1:	fprem
	fstsw	%ax
	and	$04,%ah
	jnz	1b
	fstp	%st(1)
	ret
END (__ieee754_fmodl)
libm_alias_finite (__ieee754_fmodl, __fmodl)

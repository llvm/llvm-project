/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Changes for long double by Ulrich Drepper <drepper@cygnus.com>
 * Changes for x86-64 by Andreas Jaeger <aj@suse.de>=09
 * Public domain.
 */

#include <machine/asm.h>

ENTRY(__scalbnl)
	movl	%edi,-4(%rsp)
	fildl	-4(%rsp)
	fldt	8(%rsp)
	fscale
	fstp	%st(1)
	ret
END (__scalbnl)

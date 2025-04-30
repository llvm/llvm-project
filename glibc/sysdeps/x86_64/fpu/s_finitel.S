/*
 * Written by Joe Keane <jgk@jgk.org>.
 * Adopted for x86-64 by Andreas Jaeger <aj@suse.de>.
 */

#include <machine/asm.h>

ENTRY(__finitel)
	movl	16(%rsp),%eax
	orl	$0xffff8000, %eax
	incl	%eax
	shrl	$31, %eax
	ret
END (__finitel)
weak_alias (__finitel, finitel)
hidden_def (__finitel)

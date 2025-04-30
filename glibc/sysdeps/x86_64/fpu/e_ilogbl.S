/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Changes for long double by Ulrich Drepper <drepper@cygnus.com>
 * Adapted for x86-64 by Andreas Jaeger <aj@suse.de>.
 * Public domain.
 */

#include <machine/asm.h>

ENTRY(__ieee754_ilogbl)
	fldt	8(%rsp)
/* I added the following ugly construct because ilogb(+-Inf) is
   required to return INT_MAX in ISO C99.
   -- jakub@redhat.com.  */
	fxam			/* Is NaN or +-Inf?  */
	fstsw   %ax
	movb    $0x45, %dh
	andb    %ah, %dh
	cmpb    $0x05, %dh
	je      1f		/* Is +-Inf, jump.  */
	cmpb    $0x40, %dh
	je      2f		/* Is +-Inf, jump.  */

	fxtract
	fstp	%st

	fistpl	-4(%rsp)
	fwait
	movl	-4(%rsp),%eax

	ret

1:	fstp	%st
	movl	$0x7fffffff, %eax
	ret
2:	fstp	%st
	movl	$0x80000000, %eax	/* FP_ILOGB0  */
	ret
END (__ieee754_ilogbl)

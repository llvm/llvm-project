/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>

RCSID("$NetBSD: s_ilogb.S,v 1.5 1995/10/12 15:53:09 jtc Exp $")

ENTRY(__ieee754_ilogb)
	fldl	4(%esp)
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
	je      2f		/* Is +-0, jump.  */

	fxtract
	pushl	%eax
	cfi_adjust_cfa_offset (4)
	fstp	%st

	fistpl	(%esp)
	fwait
	popl	%eax
	cfi_adjust_cfa_offset (-4)

	ret

1:	fstp	%st
	movl	$0x7fffffff, %eax
	ret
2:	fstp	%st
	movl	$0x80000000, %eax	/* FP_ILOGB0  */
	ret
END (__ieee754_ilogb)

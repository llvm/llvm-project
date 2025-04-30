/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 *
 * Adapted for i686 instructions.
 */

#include <machine/asm.h>
#include <libm-alias-finite.h>

	.text
ENTRY(__ieee754_log)
	fldln2			// log(2)
	fldl	4(%esp)		// x : log(2)
	fucomi	%st
	jp	3f
	fyl2x			// log(x)
	ret

3:	fstp	%st(1)
	ret
END (__ieee754_log)

ENTRY(__log_finite)
	fldln2			// log(2)
	fldl	4(%esp)		// x : log(2)
	fyl2x			// log(x)
	ret
END(__log_finite)
libm_alias_finite (__log_finite, __log)

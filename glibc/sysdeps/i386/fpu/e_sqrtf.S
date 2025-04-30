/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <libm-alias-finite.h>

ENTRY(__ieee754_sqrtf)
	flds	4(%esp)
	fsqrt
	ret
END (__ieee754_sqrtf)
libm_alias_finite (__ieee754_sqrtf, __sqrtf)

/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */
/* Adapted for use as nearbyint by Ulrich Drepper <drepper@cygnus.com>.  */

#include <machine/asm.h>
#include <libm-alias-float.h>

ENTRY(__nearbyintf)
	flds	4(%esp)
	subl	$32, %esp
	cfi_adjust_cfa_offset (32)
	fnstenv	4(%esp)
	frndint
	fldenv	4(%esp)
	addl	$32, %esp
	cfi_adjust_cfa_offset (-32)
	ret
END (__nearbyintf)
libm_alias_float (__nearbyint, nearbyint)

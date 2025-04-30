/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#include <machine/asm.h>
#include <libm-alias-finite.h>

ENTRY(__ieee754_sqrt)
	fldl	4(%esp)
	subl	$8, %esp
	cfi_adjust_cfa_offset (8)
	fstcw	4(%esp)
	movl	$0xfeff, %edx
	andl	4(%esp), %edx
	movl	%edx, (%esp)
	fldcw	(%esp)
	fsqrt
	fldcw	4(%esp)
	addl	$8, %esp
	cfi_adjust_cfa_offset (-8)
	ret
END (__ieee754_sqrt)
libm_alias_finite (__ieee754_sqrt, __sqrt)

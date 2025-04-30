/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 *
 * Adapted for `long double' by Ulrich Drepper <drepper@cygnus.com>.
 *
 * Correct handling of y==-inf <drepper@gnu>
 */

#include <machine/asm.h>
#include <libm-alias-finite.h>

	.section .rodata

	.align ALIGNARG(4)
	.type zero_nan,@object
zero_nan:
	.double 0.0
nan:	.byte 0, 0, 0, 0, 0, 0, 0xff, 0x7f
	.byte 0, 0, 0, 0, 0, 0, 0, 0x80
	.byte 0, 0, 0, 0, 0, 0, 0xff, 0x7f
	ASM_SIZE_DIRECTIVE(zero_nan)


#ifdef PIC
# define MO(op) op##@GOTOFF(%ecx)
# define MOX(op,x,f) op##@GOTOFF(%ecx,x,f)
#else
# define MO(op) op
# define MOX(op,x,f) op(,x,f)
#endif

	.text
ENTRY(__ieee754_scalbl)
	fldt	16(%esp)
	fxam
	fnstsw
	fldt	4(%esp)
	andl	$0x4700, %eax
	cmpl	$0x0700, %eax
	je	1f
	andl	$0x4500, %eax
	cmpl	$0x0100, %eax
	je	2f
	fxam
	fnstsw
	andl	$0x4500, %eax
	cmpl	$0x0100, %eax
	je	2f
	fld	%st(1)
	frndint
	fcomp	%st(2)
	fnstsw
	sahf
	jne	4f
	fscale
	fstp	%st(1)
	ret

	/* y is -inf */
1:	fxam
#ifdef  PIC
	LOAD_PIC_REG (cx)
#endif
	fnstsw
	movl	12(%esp), %edx
	shrl	$5, %eax
	fstp	%st
	fstp	%st
	andl	$0x8000, %edx
	andl	$0x0228, %eax
	cmpl	$0x0028, %eax
	je	4f
	andl	$8, %eax
	shrl	$11, %edx
	addl	%edx, %eax
	fldl	MOX(zero_nan, %eax, 1)
	ret

	/* The result is NaN; raise an exception for sNaN arguments.  */
2:	faddp
	ret

	/* Return NaN and raise the invalid exception.  */
4:	fstp	%st
	fstp	%st
	fldz
	fdiv	%st
	ret
END(__ieee754_scalbl)
libm_alias_finite (__ieee754_scalbl, __scalbl)

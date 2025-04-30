/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 *
 * Correct handling of y==-inf <drepper@gnu>
 */

#include <machine/asm.h>
#include <i386-math-asm.h>
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
ENTRY(__ieee754_scalb)
	fldl	12(%esp)
	fxam
	fnstsw
	fldl	4(%esp)
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
	je	3f
	fld	%st(1)
	frndint
	fcomp	%st(2)
	fnstsw
	sahf
	jne	4f
	fscale
	fstp	%st(1)
	DBL_NARROW_EVAL
	ret

	/* y is -inf */
1:	fxam
#ifdef  PIC
	LOAD_PIC_REG (cx)
#endif
	fnstsw
	movl	8(%esp), %edx
	shrl	$5, %eax
	fstp	%st
	fstp	%st
	andl	$0x80000000, %edx
	andl	$0x0228, %eax
	cmpl	$0x0028, %eax
	je	4f
	andl	$8, %eax
	shrl	$27, %edx
	addl	%edx, %eax
	fldl	MOX(zero_nan, %eax, 1)
	ret

	/* The result is NaN, but we must not raise an exception.
	   So use a variable.  */
2:	fstp	%st
	fstp	%st
#ifdef  PIC
	LOAD_PIC_REG (cx)
#endif
	fldl	MO(nan)
	ret

	/* The first parameter is a NaN.  Return it.  */
3:	fstp	%st(1)
	ret

	/* Return NaN and raise the invalid exception.  */
4:	fstp	%st
	fstp	%st
	fldz
	fdiv	%st
	ret
END(__ieee754_scalb)
libm_alias_finite (__ieee754_scalb, __scalb)

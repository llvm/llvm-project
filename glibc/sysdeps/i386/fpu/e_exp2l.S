/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Adapted for exp2 by Ulrich Drepper <drepper@cygnus.com>.
 * Public domain.
 */

#include <machine/asm.h>
#include <i386-math-asm.h>
#include <libm-alias-finite.h>

DEFINE_LDBL_MIN

#ifdef PIC
# define MO(op) op##@GOTOFF(%ecx)
#else
# define MO(op) op
#endif

	.text
ENTRY(__ieee754_exp2l)
#ifdef PIC
	LOAD_PIC_REG (cx)
#endif
	fldt	4(%esp)
/* I added the following ugly construct because exp(+-Inf) resulted
   in NaN.  The ugliness results from the bright minds at Intel.
   For the i686 the code can be written better.
   -- drepper@cygnus.com.  */
	fxam				/* Is NaN or +-Inf?  */
	fstsw	%ax
	movb	$0x45, %dh
	andb	%ah, %dh
	cmpb	$0x05, %dh
	je	1f			/* Is +-Inf, jump.  */
	movzwl	4+8(%esp), %eax
	andl	$0x7fff, %eax
	cmpl	$0x3fbe, %eax
	jge	3f
	/* Argument's exponent below -65, result rounds to 1.  */
	fld1
	faddp
	ret
3:	fld	%st
	frndint				/* int(x) */
	fsubr	%st,%st(1)		/* fract(x) */
	fxch
	f2xm1				/* 2^(fract(x)) - 1 */
	fld1
	faddp				/* 2^(fract(x)) */
	fscale				/* e^x */
	fstp	%st(1)
	LDBL_CHECK_FORCE_UFLOW_NONNEG_NAN
	ret

1:	testl	$0x200, %eax		/* Test sign.  */
	jz	2f			/* If positive, jump.  */
	fstp	%st
	fldz				/* Set result to 0.  */
2:	ret
END (__ieee754_exp2l)
libm_alias_finite (__ieee754_exp2l, __exp2l)

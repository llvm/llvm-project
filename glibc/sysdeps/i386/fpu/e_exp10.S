/*
 * Written by Ulrich Drepper <drepper@cygnus.com>.
 */

#include <machine/asm.h>
#include <i386-math-asm.h>
#include <libm-alias-finite.h>

DEFINE_DBL_MIN

#ifdef PIC
# define MO(op) op##@GOTOFF(%ecx)
#else
# define MO(op) op
#endif

	.text
/* 10^x = 2^(x * log2(10)) */
ENTRY(__ieee754_exp10)
#ifdef  PIC
	LOAD_PIC_REG (cx)
#endif
	fldl	4(%esp)
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
	fldl2t
	fmulp				/* x * log2(10) */
	fld	%st
	frndint				/* int(x * log2(10)) */
	fsubr	%st,%st(1)		/* fract(x * log2(10)) */
	fxch
	f2xm1				/* 2^(fract(x * log2(10))) - 1 */
	fld1
	faddp				/* 2^(fract(x * log2(10))) */
	fscale				/* e^x */
	fstp	%st(1)
	DBL_NARROW_EVAL_UFLOW_NONNEG_NAN
	ret

1:	testl	$0x200, %eax		/* Test sign.  */
	jz	2f			/* If positive, jump.  */
	fstp	%st
	fldz				/* Set result to 0.  */
2:	ret
END (__ieee754_exp10)
libm_alias_finite (__ieee754_exp10, __exp10)

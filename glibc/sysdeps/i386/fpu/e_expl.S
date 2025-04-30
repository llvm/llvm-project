/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 *
 * Adapted for `long double' by Ulrich Drepper <drepper@cygnus.com>.
 */

/*
 * The 8087 method for the exponential function is to calculate
 *   exp(x) = 2^(x log2(e))
 * after separating integer and fractional parts
 *   x log2(e) = i + f, |f| <= .5
 * 2^i is immediate but f needs to be precise for long double accuracy.
 * Suppress range reduction error in computing f by the following.
 * Separate x into integer and fractional parts
 *   x = xi + xf, |xf| <= .5
 * Separate log2(e) into the sum of an exact number c0 and small part c1.
 *   c0 + c1 = log2(e) to extra precision
 * Then
 *   f = (c0 xi - i) + c0 xf + c1 x
 * where c0 xi is exact and so also is (c0 xi - i).
 * -- moshier@na-net.ornl.gov
 */

#include <libm-alias-ldouble.h>
#include <machine/asm.h>
#include <i386-math-asm.h>
#include <libm-alias-finite.h>

#ifdef USE_AS_EXP10L
# define IEEE754_EXPL __ieee754_exp10l
# define FLDLOG fldl2t
#elif defined USE_AS_EXPM1L
# define IEEE754_EXPL __expm1l
# define FLDLOG fldl2e
#else
# define IEEE754_EXPL __ieee754_expl
# define FLDLOG fldl2e
#endif

	.section .rodata.cst16,"aM",@progbits,16

	.p2align 4
#ifdef USE_AS_EXP10L
	.type c0,@object
c0:	.byte 0, 0, 0, 0, 0, 0, 0x9a, 0xd4, 0x00, 0x40
	.byte 0, 0, 0, 0, 0, 0
	ASM_SIZE_DIRECTIVE(c0)
	.type c1,@object
c1:	.byte 0x58, 0x92, 0xfc, 0x15, 0x37, 0x9a, 0x97, 0xf0, 0xef, 0x3f
	.byte 0, 0, 0, 0, 0, 0
	ASM_SIZE_DIRECTIVE(c1)
#else
	.type c0,@object
c0:	.byte 0, 0, 0, 0, 0, 0, 0xaa, 0xb8, 0xff, 0x3f
	.byte 0, 0, 0, 0, 0, 0
	ASM_SIZE_DIRECTIVE(c0)
	.type c1,@object
c1:	.byte 0x20, 0xfa, 0xee, 0xc2, 0x5f, 0x70, 0xa5, 0xec, 0xed, 0x3f
	.byte 0, 0, 0, 0, 0, 0
	ASM_SIZE_DIRECTIVE(c1)
#endif
#ifndef USE_AS_EXPM1L
	.type csat,@object
csat:	.byte 0, 0, 0, 0, 0, 0, 0, 0x80, 0x0e, 0x40
	.byte 0, 0, 0, 0, 0, 0
	ASM_SIZE_DIRECTIVE(csat)
DEFINE_LDBL_MIN
#endif

#ifdef PIC
# define MO(op) op##@GOTOFF(%ecx)
#else
# define MO(op) op
#endif

	.text
ENTRY(IEEE754_EXPL)
#ifdef USE_AS_EXPM1L
	movzwl	4+8(%esp), %eax
	xorb	$0x80, %ah	// invert sign bit (now 1 is "positive")
	cmpl	$0xc006, %eax	// is num positive and exp >= 6 (number is >= 128.0)?
	jae	HIDDEN_JUMPTARGET (__expl) // (if num is denormal, it is at least >= 64.0)
#endif
	fldt	4(%esp)
/* I added the following ugly construct because expl(+-Inf) resulted
   in NaN.  The ugliness results from the bright minds at Intel.
   For the i686 the code can be written better.
   -- drepper@cygnus.com.  */
	fxam			/* Is NaN or +-Inf?  */
#ifdef PIC
	LOAD_PIC_REG (cx)
#endif
#ifdef USE_AS_EXPM1L
	xorb	$0x80, %ah
	cmpl	$0xc006, %eax
	fstsw	%ax
	movb	$0x45, %dh
	jb	4f

	/* Below -64.0 (may be -NaN or -Inf). */
	andb	%ah, %dh
	cmpb	$0x01, %dh
	je	6f		/* Is +-NaN, jump.  */
	jmp	1f		/* -large, possibly -Inf.  */

4:	/* In range -64.0 to 64.0 (may be +-0 but not NaN or +-Inf).  */
	/* Test for +-0 as argument.  */
	andb	%ah, %dh
	cmpb	$0x40, %dh
	je	2f

	/* Test for arguments that are small but not subnormal.  */
	movzwl	4+8(%esp), %eax
	andl	$0x7fff, %eax
	cmpl	$0x3fbf, %eax
	jge	3f
	/* Argument's exponent below -64; avoid spurious underflow if
	   normal.  */
	cmpl	$0x0001, %eax
	jge	2f
	/* Force underflow and return the argument, to avoid wrong signs
	   of zero results from the code below in some rounding modes.  */
	fld	%st
	fmul	%st
	fstp	%st
	jmp	2f
#else
	movzwl	4+8(%esp), %eax
	andl	$0x7fff, %eax
	cmpl	$0x400d, %eax
	jg	5f
	cmpl	$0x3fbc, %eax
	jge	3f
	/* Argument's exponent below -67, result rounds to 1.  */
	fld1
	faddp
	jmp	2f
5:	/* Overflow, underflow or infinity or NaN as argument.  */
	fstsw	%ax
	movb	$0x45, %dh
	andb	%ah, %dh
	cmpb	$0x05, %dh
	je	1f		/* Is +-Inf, jump.    */
	cmpb	$0x01, %dh
	je	6f		/* Is +-NaN, jump.    */
	/* Overflow or underflow; saturate.  */
	fstp	%st
	fldt	MO(csat)
	andb	$2, %ah
	jz	3f
	fchs
#endif
3:	FLDLOG			/* 1  log2(base)      */
	fmul	%st(1), %st	/* 1  x log2(base)    */
	/* Set round-to-nearest temporarily.  */
	subl	$8, %esp
	cfi_adjust_cfa_offset (8)
	fstcw	4(%esp)
	movl	$0xf3ff, %edx
	andl	4(%esp), %edx
	movl	%edx, (%esp)
	fldcw	(%esp)
	frndint			/* 1  i               */
	fld	%st(1)		/* 2  x               */
	frndint			/* 2  xi              */
	fldcw	4(%esp)
	addl	$8, %esp
	cfi_adjust_cfa_offset (-8)
	fld	%st(1)		/* 3  i               */
	fldt	MO(c0)		/* 4  c0              */
	fld	%st(2)		/* 5  xi              */
	fmul	%st(1), %st	/* 5  c0 xi           */
	fsubp	%st, %st(2)	/* 4  f = c0 xi  - i  */
	fld	%st(4)		/* 5  x               */
	fsub	%st(3), %st	/* 5  xf = x - xi     */
	fmulp	%st, %st(1)	/* 4  c0 xf           */
	faddp	%st, %st(1)	/* 3  f = f + c0 xf   */
	fldt	MO(c1)		/* 4                  */
	fmul	%st(4), %st	/* 4  c1 * x          */
	faddp	%st, %st(1)	/* 3  f = f + c1 * x  */
	f2xm1			/* 3 2^(fract(x * log2(base))) - 1 */
#ifdef USE_AS_EXPM1L
	fstp	%st(1)		/* 2                  */
	fscale			/* 2 scale factor is st(1); base^x - 2^i */
	fxch			/* 2 i                */
	fld1			/* 3 1.0              */
	fscale			/* 3 2^i              */
	fld1			/* 4 1.0              */
	fsubrp	%st, %st(1)	/* 3 2^i - 1.0        */
	fstp	%st(1)		/* 2                  */
	faddp	%st, %st(1)	/* 1 base^x - 1.0     */
#else
	fld1			/* 4 1.0              */
	faddp			/* 3 2^(fract(x * log2(base))) */
	fstp	%st(1)		/* 2  */
	fscale			/* 2 scale factor is st(1); base^x */
	fstp	%st(1)		/* 1  */
	LDBL_CHECK_FORCE_UFLOW_NONNEG
#endif
	fstp	%st(1)		/* 0  */
	jmp	2f
1:
#ifdef USE_AS_EXPM1L
	/* For expm1l, only negative sign gets here.  */
	fstp	%st
	fld1
	fchs
#else
	testl	$0x200, %eax	/* Test sign.  */
	jz	2f		/* If positive, jump.  */
	fstp	%st
	fldz			/* Set result to 0.  */
#endif
2:	ret
6:	/* NaN argument.  */
	fadd	%st
	ret
END(IEEE754_EXPL)

#ifdef USE_AS_EXPM1L
libm_hidden_def (__expm1l)
libm_alias_ldouble (__expm1, expm1)
#elif defined USE_AS_EXP10L
libm_alias_finite (__ieee754_exp10l, __exp10l)
#else
libm_alias_finite (__ieee754_expl, __expl)
#endif

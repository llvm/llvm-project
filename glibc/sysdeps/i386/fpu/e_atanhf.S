/* ix87 specific implementation of arctanh function.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <machine/asm.h>
#include <i386-math-asm.h>
#include <libm-alias-finite.h>

	.section .rodata

	.align ALIGNARG(4)
	.type half,@object
half:	.double 0.5
	ASM_SIZE_DIRECTIVE(half)
	.type one,@object
one:	.double 1.0
	ASM_SIZE_DIRECTIVE(one)
	.type limit,@object
limit:	.double 0.29
	ASM_SIZE_DIRECTIVE(limit)
	.align ALIGNARG(4)
	.type ln2_2,@object
ln2_2:	.tfloat 0.3465735902799726547086160
	ASM_SIZE_DIRECTIVE(ln2_2)

DEFINE_FLT_MIN

#ifdef PIC
#define MO(op) op##@GOTOFF(%edx)
#else
#define MO(op) op
#endif

	.text
ENTRY(__ieee754_atanhf)
	movl	4(%esp), %ecx

	movl	%ecx, %eax
	andl	$0x7fffffff, %eax
	cmpl	$0x7f800000, %eax
	ja	5f

#ifdef PIC
	LOAD_PIC_REG (dx)
#endif

	andl	$0x80000000, %ecx // ECX == 0 iff X >= 0

	fldt	MO(ln2_2)	// 0.5*ln2
	xorl	%ecx, 4(%esp)
	flds	4(%esp)		// |x| : 0.5*ln2
	fcoml	MO(half)	// |x| : 0.5*ln2
	fld	%st(0)		// |x| : |x| : 0.5*ln2
	fnstsw			// |x| : |x| : 0.5*ln2
	sahf
	jae	2f
	fadd	%st, %st(1)	// |x| : 2*|x| : 0.5*ln2
	fld	%st		// |x| : |x| : 2*|x| : 0.5*ln2
	fsubrl	MO(one)		// 1-|x| : |x| : 2*|x| : 0.5*ln2
	fxch			// |x| : 1-|x| : 2*|x| : 0.5*ln2
	fmul	%st(2)		// 2*|x|^2 : 1-|x| : 2*|x| : 0.5*ln2
	fdivp			// (2*|x|^2)/(1-|x|) : 2*|x| : 0.5*ln2
	faddp			// 2*|x|+(2*|x|^2)/(1-|x|) : 0.5*ln2
	fcoml	MO(limit)	// 2*|x|+(2*|x|^2)/(1-|x|) : 0.5*ln2
	fnstsw			// 2*|x|+(2*|x|^2)/(1-|x|) : 0.5*ln2
	sahf
	jae	4f
	fyl2xp1			// 0.5*ln2*ld(1+2*|x|+(2*|x|^2)/(1-|x|))
	FLT_CHECK_FORCE_UFLOW_NONNEG
	jecxz	3f
	fchs			// 0.5*ln2*ld(1+2*x+(2*x^2)/(1-x))
3:	ret

	.align ALIGNARG(4)
4:	faddl	MO(one)		// 1+2*|x|+(2*|x|^2)/(1-|x|) : 0.5*ln2
	fyl2x			// 0.5*ln2*ld(1+2*|x|+(2*|x|^2)/(1-|x|))
	jecxz	3f
	fchs			// 0.5*ln2*ld(1+2*x+(2*x^2)/(1-x))
3:	ret

	.align ALIGNARG(4)
2:	faddl	MO(one)		// 1+|x| : |x| : 0.5*ln2
	fxch			// |x| : 1+|x| : 0.5*ln2
	fsubrl	MO(one)		// 1-|x| : 1+|x| : 0.5*ln2
	fdivrp			// (1+|x|)/(1-|x|) : 0.5*ln2
	fyl2x			// 0.5*ln2*ld((1+|x|)/(1-|x|))
	jecxz	3f
	fchs			// 0.5*ln2*ld((1+x)/(1-x))
3:	ret

	// x == NaN
5:	flds	4(%esp)
	ret
END(__ieee754_atanhf)
libm_alias_finite (__ieee754_atanhf, __atanhf)

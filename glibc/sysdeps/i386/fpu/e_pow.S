/* ix87 specific implementation of pow function.
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

	.section .rodata.cst8,"aM",@progbits,8

	.p2align 3
	.type one,@object
one:	.double 1.0
	ASM_SIZE_DIRECTIVE(one)
	.type limit,@object
limit:	.double 0.29
	ASM_SIZE_DIRECTIVE(limit)
	.type p63,@object
p63:	.byte 0, 0, 0, 0, 0, 0, 0xe0, 0x43
	ASM_SIZE_DIRECTIVE(p63)
	.type p10,@object
p10:	.byte 0, 0, 0, 0, 0, 0, 0x90, 0x40
	ASM_SIZE_DIRECTIVE(p10)

	.section .rodata.cst16,"aM",@progbits,16

	.p2align 3
	.type infinity,@object
inf_zero:
infinity:
	.byte 0, 0, 0, 0, 0, 0, 0xf0, 0x7f
	ASM_SIZE_DIRECTIVE(infinity)
	.type zero,@object
zero:	.double 0.0
	ASM_SIZE_DIRECTIVE(zero)
	.type minf_mzero,@object
minf_mzero:
minfinity:
	.byte 0, 0, 0, 0, 0, 0, 0xf0, 0xff
mzero:
	.byte 0, 0, 0, 0, 0, 0, 0, 0x80
	ASM_SIZE_DIRECTIVE(minf_mzero)
DEFINE_DBL_MIN

#ifdef PIC
# define MO(op) op##@GOTOFF(%ecx)
# define MOX(op,x,f) op##@GOTOFF(%ecx,x,f)
#else
# define MO(op) op
# define MOX(op,x,f) op(,x,f)
#endif

	.text
ENTRY(__ieee754_pow)
	fldl	12(%esp)	// y
	fxam

#ifdef	PIC
	LOAD_PIC_REG (cx)
#endif

	fnstsw
	movb	%ah, %dl
	andb	$0x45, %ah
	cmpb	$0x40, %ah	// is y == 0 ?
	je	11f

	cmpb	$0x05, %ah	// is y == ±inf ?
	je	12f

	cmpb	$0x01, %ah	// is y == NaN ?
	je	30f

	fldl	4(%esp)		// x : y

	subl	$8,%esp
	cfi_adjust_cfa_offset (8)

	fxam
	fnstsw
	movb	%ah, %dh
	andb	$0x45, %ah
	cmpb	$0x40, %ah
	je	20f		// x is ±0

	cmpb	$0x05, %ah
	je	15f		// x is ±inf

	cmpb	$0x01, %ah
	je	32f		// x is NaN

	fxch			// y : x

	/* fistpll raises invalid exception for |y| >= 1L<<63.  */
	fld	%st		// y : y : x
	fabs			// |y| : y : x
	fcompl	MO(p63)		// y : x
	fnstsw
	sahf
	jnc	2f

	/* First see whether `y' is a natural number.  In this case we
	   can use a more precise algorithm.  */
	fld	%st		// y : y : x
	fistpll	(%esp)		// y : x
	fildll	(%esp)		// int(y) : y : x
	fucomp	%st(1)		// y : x
	fnstsw
	sahf
	jne	3f

	/* OK, we have an integer value for y.  If large enough that
	   errors may propagate out of the 11 bits excess precision, use
	   the algorithm for real exponent instead.  */
	fld	%st		// y : y : x
	fabs			// |y| : y : x
	fcompl	MO(p10)		// y : x
	fnstsw
	sahf
	jnc	2f
	popl	%eax
	cfi_adjust_cfa_offset (-4)
	popl	%edx
	cfi_adjust_cfa_offset (-4)
	orl	$0, %edx
	fstp	%st(0)		// x
	jns	4f		// y >= 0, jump
	fdivrl	MO(one)		// 1/x		(now referred to as x)
	negl	%eax
	adcl	$0, %edx
	negl	%edx
4:	fldl	MO(one)		// 1 : x
	fxch

	/* If y is even, take the absolute value of x.  Otherwise,
	   ensure all intermediate values that might overflow have the
	   sign of x.  */
	testb	$1, %al
	jnz	6f
	fabs

6:	shrdl	$1, %edx, %eax
	jnc	5f
	fxch
	fabs
	fmul	%st(1)		// x : ST*x
	fxch
5:	fld	%st		// x : x : ST*x
	fabs			// |x| : x : ST*x
	fmulp			// |x|*x : ST*x
	shrl	$1, %edx
	movl	%eax, %ecx
	orl	%edx, %ecx
	jnz	6b
	fstp	%st(0)		// ST*x
#ifdef	PIC
	LOAD_PIC_REG (cx)
#endif
	DBL_NARROW_EVAL_UFLOW_NONNAN
	ret

	/* y is ±NAN */
30:	fldl	4(%esp)		// x : y
	fldl	MO(one)		// 1.0 : x : y
	fucomp	%st(1)		// x : y
	fnstsw
	sahf
	je	31f
	fxch			// y : x
31:	fstp	%st(1)
	ret

	cfi_adjust_cfa_offset (8)
32:	addl	$8, %esp
	cfi_adjust_cfa_offset (-8)
	fstp	%st(1)
	ret

	cfi_adjust_cfa_offset (8)
	.align ALIGNARG(4)
2:	// y is a large integer (absolute value at least 1L<<10), but
	// may be odd unless at least 1L<<64.  So it may be necessary
	// to adjust the sign of a negative result afterwards.
	fxch			// x : y
	fabs			// |x| : y
	fxch			// y : x
	.align ALIGNARG(4)
3:	/* y is a real number.  */
	fxch			// x : y
	fldl	MO(one)		// 1.0 : x : y
	fldl	MO(limit)	// 0.29 : 1.0 : x : y
	fld	%st(2)		// x : 0.29 : 1.0 : x : y
	fsub	%st(2)		// x-1 : 0.29 : 1.0 : x : y
	fabs			// |x-1| : 0.29 : 1.0 : x : y
	fucompp			// 1.0 : x : y
	fnstsw
	fxch			// x : 1.0 : y
	sahf
	ja	7f
	fsub	%st(1)		// x-1 : 1.0 : y
	fyl2xp1			// log2(x) : y
	jmp	8f

7:	fyl2x			// log2(x) : y
8:	fmul	%st(1)		// y*log2(x) : y
	fst	%st(1)		// y*log2(x) : y*log2(x)
	frndint			// int(y*log2(x)) : y*log2(x)
	fsubr	%st, %st(1)	// int(y*log2(x)) : fract(y*log2(x))
	fxch			// fract(y*log2(x)) : int(y*log2(x))
	f2xm1			// 2^fract(y*log2(x))-1 : int(y*log2(x))
	faddl	MO(one)		// 2^fract(y*log2(x)) : int(y*log2(x))

	// Before scaling, we must negate if x is negative and y is an
	// odd integer.
	testb	$2, %dh
	jz	291f
	// x is negative.  If y is an odd integer, negate the result.
	fldl	20(%esp)	// y : 2^fract(y*log2(x)) : int(y*log2(x))
	fld	%st		// y : y : 2^fract(y*log2(x)) : int(y*log2(x))
	fabs			// |y| : y : 2^fract(y*log2(x)) : int(y*log2(x))
	fcompl	MO(p63)		// y : 2^fract(y*log2(x)) : int(y*log2(x))
	fnstsw
	sahf
	jnc	290f

	// We must find out whether y is an odd integer.
	fld	%st		// y : y : 2^fract(y*log2(x)) : int(y*log2(x))
	fistpll	(%esp)		// y : 2^fract(y*log2(x)) : int(y*log2(x))
	fildll	(%esp)		// int(y) : y : 2^fract(y*log2(x)) : int(y*log2(x))
	fucompp			// 2^fract(y*log2(x)) : int(y*log2(x))
	fnstsw
	sahf
	jne	291f

	// OK, the value is an integer, but is it odd?
	popl	%eax
	cfi_adjust_cfa_offset (-4)
	popl	%edx
	cfi_adjust_cfa_offset (-4)
	andb	$1, %al
	jz	292f		// jump if not odd
	// It's an odd integer.
	fchs
	jmp	292f

	cfi_adjust_cfa_offset (8)
290:	fstp	%st(0)		// 2^fract(y*log2(x)) : int(y*log2(x))
291:	addl	$8, %esp
	cfi_adjust_cfa_offset (-8)
292:	fscale			// +/- 2^fract(y*log2(x))*2^int(y*log2(x)) : int(y*log2(x))
	fstp	%st(1)		// +/- 2^fract(y*log2(x))*2^int(y*log2(x))
	DBL_NARROW_EVAL_UFLOW_NONNAN
	ret


	// pow(x,±0) = 1
	.align ALIGNARG(4)
11:	fstp	%st(0)		// pop y
	fldl	MO(one)
	ret

	// y == ±inf
	.align ALIGNARG(4)
12:	fstp	%st(0)		// pop y
	fldl	MO(one)		// 1
	fldl	4(%esp)		// x : 1
	fabs			// abs(x) : 1
	fucompp			// < 1, == 1, or > 1
	fnstsw
	andb	$0x45, %ah
	cmpb	$0x45, %ah
	je	13f		// jump if x is NaN

	cmpb	$0x40, %ah
	je	14f		// jump if |x| == 1

	shlb	$1, %ah
	xorb	%ah, %dl
	andl	$2, %edx
	fldl	MOX(inf_zero, %edx, 4)
	ret

	.align ALIGNARG(4)
14:	fldl	MO(one)
	ret

	.align ALIGNARG(4)
13:	fldl	4(%esp)		// load x == NaN
	ret

	cfi_adjust_cfa_offset (8)
	.align ALIGNARG(4)
	// x is ±inf
15:	fstp	%st(0)		// y
	testb	$2, %dh
	jz	16f		// jump if x == +inf

	// fistpll raises invalid exception for |y| >= 1L<<63, so test
	// that (in which case y is certainly even) before testing
	// whether y is odd.
	fld	%st		// y : y
	fabs			// |y| : y
	fcompl	MO(p63)		// y
	fnstsw
	sahf
	jnc	16f

	// We must find out whether y is an odd integer.
	fld	%st		// y : y
	fistpll	(%esp)		// y
	fildll	(%esp)		// int(y) : y
	fucompp			// <empty>
	fnstsw
	sahf
	jne	17f

	// OK, the value is an integer.
	popl	%eax
	cfi_adjust_cfa_offset (-4)
	popl	%edx
	cfi_adjust_cfa_offset (-4)
	andb	$1, %al
	jz	18f		// jump if not odd
	// It's an odd integer.
	shrl	$31, %edx
	fldl	MOX(minf_mzero, %edx, 8)
	ret

	cfi_adjust_cfa_offset (8)
	.align ALIGNARG(4)
16:	fcompl	MO(zero)
	addl	$8, %esp
	cfi_adjust_cfa_offset (-8)
	fnstsw
	shrl	$5, %eax
	andl	$8, %eax
	fldl	MOX(inf_zero, %eax, 1)
	ret

	cfi_adjust_cfa_offset (8)
	.align ALIGNARG(4)
17:	shll	$30, %edx	// sign bit for y in right position
	addl	$8, %esp
	cfi_adjust_cfa_offset (-8)
18:	shrl	$31, %edx
	fldl	MOX(inf_zero, %edx, 8)
	ret

	cfi_adjust_cfa_offset (8)
	.align ALIGNARG(4)
	// x is ±0
20:	fstp	%st(0)		// y
	testb	$2, %dl
	jz	21f		// y > 0

	// x is ±0 and y is < 0.  We must find out whether y is an odd integer.
	testb	$2, %dh
	jz	25f

	// fistpll raises invalid exception for |y| >= 1L<<63, so test
	// that (in which case y is certainly even) before testing
	// whether y is odd.
	fld	%st		// y : y
	fabs			// |y| : y
	fcompl	MO(p63)		// y
	fnstsw
	sahf
	jnc	25f

	fld	%st		// y : y
	fistpll	(%esp)		// y
	fildll	(%esp)		// int(y) : y
	fucompp			// <empty>
	fnstsw
	sahf
	jne	26f

	// OK, the value is an integer.
	popl	%eax
	cfi_adjust_cfa_offset (-4)
	popl	%edx
	cfi_adjust_cfa_offset (-4)
	andb	$1, %al
	jz	27f		// jump if not odd
	// It's an odd integer.
	// Raise divide-by-zero exception and get minus infinity value.
	fldl	MO(one)
	fdivl	MO(zero)
	fchs
	ret

	cfi_adjust_cfa_offset (8)
25:	fstp	%st(0)
26:	addl	$8, %esp
	cfi_adjust_cfa_offset (-8)
27:	// Raise divide-by-zero exception and get infinity value.
	fldl	MO(one)
	fdivl	MO(zero)
	ret

	cfi_adjust_cfa_offset (8)
	.align ALIGNARG(4)
	// x is ±0 and y is > 0.  We must find out whether y is an odd integer.
21:	testb	$2, %dh
	jz	22f

	// fistpll raises invalid exception for |y| >= 1L<<63, so test
	// that (in which case y is certainly even) before testing
	// whether y is odd.
	fcoml	MO(p63)		// y
	fnstsw
	sahf
	jnc	22f

	fld	%st		// y : y
	fistpll	(%esp)		// y
	fildll	(%esp)		// int(y) : y
	fucompp			// <empty>
	fnstsw
	sahf
	jne	23f

	// OK, the value is an integer.
	popl	%eax
	cfi_adjust_cfa_offset (-4)
	popl	%edx
	cfi_adjust_cfa_offset (-4)
	andb	$1, %al
	jz	24f		// jump if not odd
	// It's an odd integer.
	fldl	MO(mzero)
	ret

	cfi_adjust_cfa_offset (8)
22:	fstp	%st(0)
23:	addl	$8, %esp	// Don't use 2 x pop
	cfi_adjust_cfa_offset (-8)
24:	fldl	MO(zero)
	ret

END(__ieee754_pow)
libm_alias_finite (__ieee754_pow, __pow)

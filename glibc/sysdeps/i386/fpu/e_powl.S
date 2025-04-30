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
	.type p2,@object
p2:	.byte 0, 0, 0, 0, 0, 0, 0x10, 0x40
	ASM_SIZE_DIRECTIVE(p2)
	.type p63,@object
p63:	.byte 0, 0, 0, 0, 0, 0, 0xe0, 0x43
	ASM_SIZE_DIRECTIVE(p63)
	.type p64,@object
p64:	.byte 0, 0, 0, 0, 0, 0, 0xf0, 0x43
	ASM_SIZE_DIRECTIVE(p64)
	.type p78,@object
p78:	.byte 0, 0, 0, 0, 0, 0, 0xd0, 0x44
	ASM_SIZE_DIRECTIVE(p78)
	.type pm79,@object
pm79:	.byte 0, 0, 0, 0, 0, 0, 0, 0x3b
	ASM_SIZE_DIRECTIVE(pm79)

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
DEFINE_LDBL_MIN

#ifdef PIC
# define MO(op) op##@GOTOFF(%ecx)
# define MOX(op,x,f) op##@GOTOFF(%ecx,x,f)
#else
# define MO(op) op
# define MOX(op,x,f) op(,x,f)
#endif

	.text
ENTRY(__ieee754_powl)
	fldt	16(%esp)	// y
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

	fldt	4(%esp)		// x : y

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
	je	9f

	// If y has absolute value at most 0x1p-79, then any finite
	// nonzero x will result in 1.  Saturate y to those bounds to
	// avoid underflow in the calculation of y*log2(x).
	fld	%st		// y : y : x
	fabs			// |y| : y : x
	fcompl	MO(pm79)	// y : x
	fnstsw
	sahf
	jnc	3f
	fstp	%st(0)		// pop y
	fldl	MO(pm79)	// 0x1p-79 : x
	testb	$2, %dl
	jnz	3f		// y > 0
	fchs			// -0x1p-79 : x
	jmp	3f

9:	/* OK, we have an integer value for y.  Unless very small
	   (we use < 4), use the algorithm for real exponent to avoid
	   accumulation of errors.  */
	fld	%st		// y : y : x
	fabs			// |y| : y : x
	fcompl	MO(p2)		// y : x
	fnstsw
	sahf
	jnc	3f
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
	LDBL_CHECK_FORCE_UFLOW_NONNAN
	ret

	/* y is ±NAN */
30:	fldt	4(%esp)		// x : y
	fldl	MO(one)		// 1.0 : x : y
	fucomp	%st(1)		// x : y
	fnstsw
	sahf
	je	33f
31:	/* At least one argument NaN, and result should be NaN.  */
	faddp
	ret
33:	jp	31b
	/* pow (1, NaN); check if the NaN signaling.  */
	testb	$0x40, 23(%esp)
	jz	31b
	fstp	%st(1)
	ret

	cfi_adjust_cfa_offset (8)
32:	addl	$8, %esp
	cfi_adjust_cfa_offset (-8)
	faddp
	ret

	cfi_adjust_cfa_offset (8)
	.align ALIGNARG(4)
2:	// y is a large integer (absolute value at least 1L<<63).
	// If y has absolute value at least 1L<<78, then any finite
	// nonzero x will result in 0 (underflow), 1 or infinity (overflow).
	// Saturate y to those bounds to avoid overflow in the calculation
	// of y*log2(x).
	fld	%st		// y : y : x
	fabs			// |y| : y : x
	fcompl	MO(p78)		// y : x
	fnstsw
	sahf
	jc	3f
	fstp	%st(0)		// pop y
	fldl	MO(p78)		// 1L<<78 : x
	testb	$2, %dl
	jz	3f		// y > 0
	fchs			// -(1L<<78) : x
	.align ALIGNARG(4)
3:	/* y is a real number.  */
	subl	$28, %esp
	cfi_adjust_cfa_offset (28)
	fstpt	12(%esp)	// x
	fstpt	(%esp)		// <empty>
	call	HIDDEN_JUMPTARGET (__powl_helper)	// <result>
	addl	$36, %esp
	cfi_adjust_cfa_offset (-36)
	ret

	// pow(x,±0) = 1, unless x is sNaN
	.align ALIGNARG(4)
11:	fstp	%st(0)		// pop y
	fldt	4(%esp)		// x
	fxam
	fnstsw
	andb	$0x45, %ah
	cmpb	$0x01, %ah
	je	112f		// x is NaN
111:	fstp	%st(0)
	fldl	MO(one)
	ret

112:	testb	$0x40, 11(%esp)
	jnz	111b
	fadd	%st(0)
	ret

	// y == ±inf
	.align ALIGNARG(4)
12:	fstp	%st(0)		// pop y
	fldl	MO(one)		// 1
	fldt	4(%esp)		// x : 1
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
13:	fldt	4(%esp)		// load x == NaN
	fadd	%st(0)
	ret

	cfi_adjust_cfa_offset (8)
	.align ALIGNARG(4)
	// x is ±inf
15:	fstp	%st(0)		// y
	testb	$2, %dh
	jz	16f		// jump if x == +inf

	// fistpll raises invalid exception for |y| >= 1L<<63, but y
	// may be odd unless we know |y| >= 1L<<64.
	fld	%st		// y : y
	fabs			// |y| : y
	fcompl	MO(p64)		// y
	fnstsw
	sahf
	jnc	16f
	fldl	MO(p63)		// p63 : y
	fxch			// y : p63
	fprem			// y%p63 : p63
	fstp	%st(1)		// y%p63

	// We must find out whether y is an odd integer.
	fld	%st		// y : y
	fistpll	(%esp)		// y
	fildll	(%esp)		// int(y) : y
	fucompp			// <empty>
	fnstsw
	sahf
	jne	17f

	// OK, the value is an integer, but is it odd?
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

	// fistpll raises invalid exception for |y| >= 1L<<63, but y
	// may be odd unless we know |y| >= 1L<<64.
	fld	%st		// y : y
	fabs			// |y| : y
	fcompl	MO(p64)		// y
	fnstsw
	sahf
	jnc	25f
	fldl	MO(p63)		// p63 : y
	fxch			// y : p63
	fprem			// y%p63 : p63
	fstp	%st(1)		// y%p63

	fld	%st		// y : y
	fistpll	(%esp)		// y
	fildll	(%esp)		// int(y) : y
	fucompp			// <empty>
	fnstsw
	sahf
	jne	26f

	// OK, the value is an integer, but is it odd?
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

	// fistpll raises invalid exception for |y| >= 1L<<63, but y
	// may be odd unless we know |y| >= 1L<<64.
	fld	%st		// y : y
	fcompl	MO(p64)		// y
	fnstsw
	sahf
	jnc	22f
	fldl	MO(p63)		// p63 : y
	fxch			// y : p63
	fprem			// y%p63 : p63
	fstp	%st(1)		// y%p63

	fld	%st		// y : y
	fistpll	(%esp)		// y
	fildll	(%esp)		// int(y) : y
	fucompp			// <empty>
	fnstsw
	sahf
	jne	23f

	// OK, the value is an integer, but is it odd?
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

END(__ieee754_powl)
libm_alias_finite (__ieee754_powl, __powl)

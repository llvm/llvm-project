/* Compute cubic root of double value.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Dirk Alboth <dirka@uni-paderborn.de> and
   Ulrich Drepper <drepper@cygnus.com>, 1997.

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
#include <libm-alias-double.h>

        .section .rodata

        .align ALIGNARG(4)
        .type f7,@object
f7:	.double -0.145263899385486377
	ASM_SIZE_DIRECTIVE(f7)
        .type f6,@object
f6:	.double 0.784932344976639262
	ASM_SIZE_DIRECTIVE(f6)
        .type f5,@object
f5:	.double -1.83469277483613086
	ASM_SIZE_DIRECTIVE(f5)
        .type f4,@object
f4:	.double 2.44693122563534430
	ASM_SIZE_DIRECTIVE(f4)
        .type f3,@object
f3:	.double -2.11499494167371287
	ASM_SIZE_DIRECTIVE(f3)
        .type f2,@object
f2:	.double 1.50819193781584896
	ASM_SIZE_DIRECTIVE(f2)
        .type f1,@object
f1:	.double 0.354895765043919860
	ASM_SIZE_DIRECTIVE(f1)

#define CBRT2		1.2599210498948731648
#define ONE_CBRT2	0.793700525984099737355196796584
#define SQR_CBRT2	1.5874010519681994748
#define ONE_SQR_CBRT2	0.629960524947436582364439673883

	.type factor,@object
factor:	.double ONE_SQR_CBRT2
	.double ONE_CBRT2
	.double 1.0
	.double CBRT2
	.double SQR_CBRT2
	ASM_SIZE_DIRECTIVE(factor)

        .type two54,@object
two54:  .byte 0, 0, 0, 0, 0, 0, 0x50, 0x43
        ASM_SIZE_DIRECTIVE(two54)

#ifdef PIC
#define MO(op) op##@GOTOFF(%ebx)
#define MOX(op,x) op##@GOTOFF(%ebx,x,1)
#else
#define MO(op) op
#define MOX(op,x) op(x)
#endif

	.text
ENTRY(__cbrt)
	movl	4(%esp), %ecx
	movl	8(%esp), %eax
	movl	%eax, %edx
	andl	$0x7fffffff, %eax
	orl	%eax, %ecx
	jz	1f
	xorl	%ecx, %ecx
	cmpl	$0x7ff00000, %eax
	jae	1f

#ifdef PIC
	pushl	%ebx
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (ebx, 0)
	LOAD_PIC_REG (bx)
#endif

	cmpl	$0x00100000, %eax
	jae	2f

#ifdef PIC
	fldl	8(%esp)
#else
	fldl	4(%esp)
#endif
	fmull	MO(two54)
	movl	$-54, %ecx
#ifdef PIC
	fstpl	8(%esp)
	movl	12(%esp), %eax
#else
	fstpl	4(%esp)
	movl	8(%esp), %eax
#endif
	movl	%eax, %edx
	andl	$0x7fffffff, %eax

2:	shrl	$20, %eax
	andl	$0x800fffff, %edx
	subl	$1022, %eax
	orl	$0x3fe00000, %edx
	addl	%eax, %ecx
#ifdef PIC
	movl	%edx, 12(%esp)

	fldl	8(%esp)			/* xm */
#else
	movl	%edx, 8(%esp)

	fldl	4(%esp)			/* xm */
#endif
	fabs

	/* The following code has two tracks:
	    a) compute the normalized cbrt value
	    b) compute xe/3 and xe%3
	   The right track computes the value for b) and this is done
	   in an optimized way by avoiding division.

	   But why two tracks at all?  Very easy: efficiency.  Some FP
	   instruction can overlap with a certain amount of integer (and
	   FP) instructions.  So we get (except for the imull) all
	   instructions for free.  */

	fld	%st(0)			/* xm : xm */

	fmull	MO(f7)			/* f7*xm : xm */
			movl	$1431655766, %eax
	faddl	MO(f6)			/* f6+f7*xm : xm */
			imull	%ecx
	fmul	%st(1)			/* (f6+f7*xm)*xm : xm */
			movl	%ecx, %eax
	faddl	MO(f5)			/* f5+(f6+f7*xm)*xm : xm */
			sarl	$31, %eax
	fmul	%st(1)			/* (f5+(f6+f7*xm)*xm)*xm : xm */
			subl	%eax, %edx
	faddl	MO(f4)			/* f4+(f5+(f6+f7*xm)*xm)*xm : xm */
	fmul	%st(1)			/* (f4+(f5+(f6+f7*xm)*xm)*xm)*xm : xm */
	faddl	MO(f3)			/* f3+(f4+(f5+(f6+f7*xm)*xm)*xm)*xm : xm */
	fmul	%st(1)			/* (f3+(f4+(f5+(f6+f7*xm)*xm)*xm)*xm)*xm : xm */
	faddl	MO(f2)			/* f2+(f3+(f4+(f5+(f6+f7*xm)*xm)*xm)*xm)*xm : xm */
	fmul	%st(1)			/* (f2+(f3+(f4+(f5+(f6+f7*xm)*xm)*xm)*xm)*xm)*xm : xm */
	faddl	MO(f1)			/* u:=f1+(f2+(f3+(f4+(f5+(f6+f7*xm)*xm)*xm)*xm)*xm)*xm : xm */

	fld	%st			/* u : u : xm */
	fmul	%st(1)			/* u*u : u : xm */
	fld	%st(2)			/* xm : u*u : u : xm */
	fadd	%st			/* 2*xm : u*u : u : xm */
	fxch	%st(1)			/* u*u : 2*xm : u : xm */
	fmul	%st(2)			/* t2:=u*u*u : 2*xm : u : xm */
			movl	%edx, %eax
	fadd	%st, %st(1)		/* t2 : t2+2*xm : u : xm */
			leal	(%edx,%edx,2),%edx
	fadd	%st(0)			/* 2*t2 : t2+2*xm : u : xm */
			subl	%edx, %ecx
	faddp	%st, %st(3)		/* t2+2*xm : u : 2*t2+xm */
			shll	$3, %ecx
	fmulp				/* u*(t2+2*xm) : 2*t2+xm */
	fdivp	%st, %st(1)		/* u*(t2+2*xm)/(2*t2+xm) */
	fmull	MOX(16+factor,%ecx)	/* u*(t2+2*xm)/(2*t2+xm)*FACT */
	pushl	%eax
	cfi_adjust_cfa_offset (4)
	fildl	(%esp)			/* xe/3 : u*(t2+2*xm)/(2*t2+xm)*FACT */
	fxch				/* u*(t2+2*xm)/(2*t2+xm)*FACT : xe/3 */
	fscale				/* u*(t2+2*xm)/(2*t2+xm)*FACT*2^xe/3 */
	popl	%edx
	cfi_adjust_cfa_offset (-4)
#ifdef PIC
	movl	12(%esp), %eax
	popl	%ebx
	cfi_adjust_cfa_offset (-4)
	cfi_restore (ebx)
#else
	movl	8(%esp), %eax
#endif
	testl	%eax, %eax
	fstp	%st(1)
	jns	4f
	fchs
4:	ret

	/* Return the argument.  */
1:	fldl	4(%esp)
	ret
END(__cbrt)
libm_alias_double (__cbrt, cbrt)

/* Compute cubic root of long double value.
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

#include <libm-alias-ldouble.h>
#include <machine/asm.h>

        .section .rodata

        .align ALIGNARG(4)
        .type f8,@object
f8:	.tfloat 0.161617097923756032
	ASM_SIZE_DIRECTIVE(f8)
        .align ALIGNARG(4)
        .type f7,@object
f7:	.tfloat -0.988553671195413709
	ASM_SIZE_DIRECTIVE(f7)
        .align ALIGNARG(4)
        .type f6,@object
f6:	.tfloat 2.65298938441952296
	ASM_SIZE_DIRECTIVE(f6)
        .align ALIGNARG(4)
        .type f5,@object
f5:	.tfloat -4.11151425200350531
	ASM_SIZE_DIRECTIVE(f5)
        .align ALIGNARG(4)
        .type f4,@object
f4:	.tfloat 4.09559907378707839
	ASM_SIZE_DIRECTIVE(f4)
        .align ALIGNARG(4)
        .type f3,@object
f3:	.tfloat -2.82414939754975962
	ASM_SIZE_DIRECTIVE(f3)
        .align ALIGNARG(4)
        .type f2,@object
f2:	.tfloat 1.67595307700780102
	ASM_SIZE_DIRECTIVE(f2)
        .align ALIGNARG(4)
        .type f1,@object
f1:	.tfloat 0.338058687610520237
	ASM_SIZE_DIRECTIVE(f1)

#define CBRT2		1.2599210498948731648
#define ONE_CBRT2	0.793700525984099737355196796584
#define SQR_CBRT2	1.5874010519681994748
#define ONE_SQR_CBRT2	0.629960524947436582364439673883

	/* We make the entries in the following table all 16 bytes
	   wide to avoid having to implement a multiplication by 10.  */
	.type factor,@object
        .align ALIGNARG(4)
factor:	.tfloat ONE_SQR_CBRT2
	.byte 0, 0, 0, 0, 0, 0
	.tfloat ONE_CBRT2
	.byte 0, 0, 0, 0, 0, 0
	.tfloat 1.0
	.byte 0, 0, 0, 0, 0, 0
	.tfloat CBRT2
	.byte 0, 0, 0, 0, 0, 0
	.tfloat SQR_CBRT2
	ASM_SIZE_DIRECTIVE(factor)

        .type two64,@object
        .align ALIGNARG(4)
two64:  .byte 0, 0, 0, 0, 0, 0, 0xf0, 0x43
        ASM_SIZE_DIRECTIVE(two64)

#ifdef PIC
#define MO(op) op##@GOTOFF(%ebx)
#define MOX(op,x) op##@GOTOFF(%ebx,x,1)
#else
#define MO(op) op
#define MOX(op,x) op(x)
#endif

	.text
ENTRY(__cbrtl)
	movl	4(%esp), %ecx
	movl	12(%esp), %eax
	orl	8(%esp), %ecx
	movl	%eax, %edx
	andl	$0x7fff, %eax
	orl	%eax, %ecx
	jz	1f
	xorl	%ecx, %ecx
	cmpl	$0x7fff, %eax
	je	1f

#ifdef PIC
	pushl	%ebx
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (ebx, 0)
	LOAD_PIC_REG (bx)
#endif

	cmpl	$0, %eax
	jne	2f

#ifdef PIC
	fldt	8(%esp)
#else
	fldt	4(%esp)
#endif
	fmull	MO(two64)
	movl	$-64, %ecx
#ifdef PIC
	fstpt	8(%esp)
	movl	16(%esp), %eax
#else
	fstpt	4(%esp)
	movl	12(%esp), %eax
#endif
	movl	%eax, %edx
	andl	$0x7fff, %eax

2:	andl	$0x8000, %edx
	subl	$16382, %eax
	orl	$0x3ffe, %edx
	addl	%eax, %ecx
#ifdef PIC
	movl	%edx, 16(%esp)

	fldt	8(%esp)			/* xm */
#else
	movl	%edx, 12(%esp)

	fldt	4(%esp)			/* xm */
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

	fldt	MO(f8)			/* f8 : xm */
	fmul	%st(1)			/* f8*xm : xm */

	fldt	MO(f7)
	faddp				/* f7+f8*xm : xm */
	fmul	%st(1)			/* (f7+f8*xm)*xm : xm */
			movl	$1431655766, %eax
	fldt	MO(f6)
	faddp				/* f6+(f7+f8*xm)*xm : xm */
			imull	%ecx
	fmul	%st(1)			/* (f6+(f7+f8*xm)*xm)*xm : xm */
			movl	%ecx, %eax
	fldt	MO(f5)
	faddp				/* f5+(f6+(f7+f8*xm)*xm)*xm : xm */
			sarl	$31, %eax
	fmul	%st(1)			/* (f5+(f6+(f7+f8*xm)*xm)*xm)*xm : xm */
			subl	%eax, %edx
	fldt	MO(f4)
	faddp				/* f4+(f5+(f6+(f7+f8*xm)*xm)*xm)*xm : xm */
	fmul	%st(1)			/* (f4+(f5+(f6+(f7+f8*xm)*xm)*xm)*xm)*xm : xm */
	fldt	MO(f3)
	faddp				/* f3+(f4+(f5+(f6+(f7+f8*xm)*xm)*xm)*xm)*xm : xm */
	fmul	%st(1)			/* (f3+(f4+(f5+(f6+(f7+f8*xm)*xm)*xm)*xm)*xm)*xm : xm */
	fldt	MO(f2)
	faddp				/* f2+(f3+(f4+(f5+(f6+(f7+f8*xm)*xm)*xm)*xm)*xm)*xm : xm */
	fmul	%st(1)			/* (f2+(f3+(f4+(f5+(f6+(f7+f8*xm)*xm)*xm)*xm)*xm)*xm)*xm : xm */
	fldt	MO(f1)
	faddp				/* u:=f1+(f2+(f3+(f4+(f5+(f6+(f7+f8*xm)*xm)*xm)*xm)*xm)*xm)*xm : xm */

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
			shll	$4, %ecx
	fmulp				/* u*(t2+2*xm) : 2*t2+xm */
	fdivp	%st, %st(1)		/* u*(t2+2*xm)/(2*t2+xm) */
	fldt	MOX(32+factor,%ecx)
	fmulp				/* u*(t2+2*xm)/(2*t2+xm)*FACT */
	pushl	%eax
	cfi_adjust_cfa_offset (4)
	fildl	(%esp)			/* xe/3 : u*(t2+2*xm)/(2*t2+xm)*FACT */
	fxch				/* u*(t2+2*xm)/(2*t2+xm)*FACT : xe/3 */
	fscale				/* u*(t2+2*xm)/(2*t2+xm)*FACT*2^xe/3 */
	popl	%edx
	cfi_adjust_cfa_offset (-4)
#ifdef PIC
	movl	16(%esp), %eax
	popl	%ebx
	cfi_adjust_cfa_offset (-4)
	cfi_restore (ebx)
#else
	movl	12(%esp), %eax
#endif
	testl	$0x8000, %eax
	fstp	%st(1)
	jz	4f
	fchs
4:	ret

	/* Return the argument.  */
1:	fldt	4(%esp)
	fadd	%st
	ret
END(__cbrtl)
libm_alias_ldouble (__cbrt, cbrt)

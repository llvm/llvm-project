/* Compute cubic root of float value.
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
#include <libm-alias-float.h>

        .section .rodata

        .align ALIGNARG(4)
        .type f3,@object
f3:	.double 0.191502161678719066
        ASM_SIZE_DIRECTIVE(f3)
        .type f2,@object
f2:	.double 0.697570460207922770
        ASM_SIZE_DIRECTIVE(f2)
        .type f1,@object
f1:	.double 0.492659620528969547
        ASM_SIZE_DIRECTIVE(f1)

#define CBRT2		1.2599210498948731648
#define ONE_CBRT2	0.793700525984099737355196796584
#define SQR_CBRT2	1.5874010519681994748
#define ONE_SQR_CBRT2	0.629960524947436582364439673883

	.type factor,@object
        .align ALIGNARG(4)
factor:	.double ONE_SQR_CBRT2
	.double ONE_CBRT2
	.double 1.0
	.double CBRT2
	.double SQR_CBRT2
	ASM_SIZE_DIRECTIVE(factor)

        .type two25,@object
two25:	.byte 0, 0, 0, 0x4c
        ASM_SIZE_DIRECTIVE(two25)

#ifdef PIC
#define MO(op) op##@GOTOFF(%ebx)
#define MOX(op,x) op##@GOTOFF(%ebx,x,1)
#else
#define MO(op) op
#define MOX(op,x) op(x)
#endif

	.text
ENTRY(__cbrtf)
	movl	4(%esp), %eax
	xorl	%ecx, %ecx
	movl	%eax, %edx
	andl	$0x7fffffff, %eax
	jz	1f
	cmpl	$0x7f800000, %eax
	jae	1f

#ifdef PIC
	pushl	%ebx
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (ebx, 0)
	LOAD_PIC_REG (bx)
#endif

	cmpl	$0x00800000, %eax
	jae	2f

#ifdef PIC
	flds	8(%esp)
#else
	flds	4(%esp)
#endif
	fmuls	MO(two25)
	movl	$-25, %ecx
#ifdef PIC
	fstps	8(%esp)
	movl	8(%esp), %eax
#else
	fstps	4(%esp)
	movl	4(%esp), %eax
#endif
	movl	%eax, %edx
	andl	$0x7fffffff, %eax

2:	shrl	$23, %eax
	andl	$0x807fffff, %edx
	subl	$126, %eax
	orl	$0x3f000000, %edx
	addl	%eax, %ecx
#ifdef PIC
	movl	%edx, 8(%esp)

	flds	8(%esp)			/* xm */
#else
	movl	%edx, 4(%esp)

	flds	4(%esp)			/* xm */
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
	fmull	MO(f3)			/* f3*xm : xm */
			movl	$1431655766, %eax
	fsubrl	MO(f2)			/* f2-f3*xm : xm */
			imull	%ecx
	fmul	%st(1)			/* (f2-f3*xm)*xm : xm */
			movl	%ecx, %eax
	faddl	MO(f1)			/* u:=f1+(f2-f3*xm)*xm : xm */
			sarl	$31, %eax
	fld	%st			/* u : u : xm */
			subl	%eax, %edx
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
	movl	8(%esp), %eax
	popl	%ebx
	cfi_adjust_cfa_offset (-4)
	cfi_restore (ebx)
#else
	movl	4(%esp), %eax
#endif
	testl	%eax, %eax
	fstp	%st(1)
	jns	4f
	fchs
4:	ret

	/* Return the argument.  */
1:	flds	4(%esp)
	ret
END(__cbrtf)
libm_alias_float (__cbrt, cbrt)

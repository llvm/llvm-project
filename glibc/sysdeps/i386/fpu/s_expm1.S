/* ix87 specific implementation of exp(x)-1.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.
   Based on code by John C. Bowman <bowman@ipp-garching.mpg.de>.
   Corrections by H.J. Lu (hjl@gnu.ai.mit.edu), 1997.

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

	/* Using: e^x - 1 = 2^(x * log2(e)) - 1 */

#include <sysdep.h>
#include <machine/asm.h>
#include <i386-math-asm.h>
#include <libm-alias-double.h>

	.section .rodata

	.align ALIGNARG(4)
	.type minus1,@object
minus1:	.double -1.0
	ASM_SIZE_DIRECTIVE(minus1)
	.type one,@object
one:	.double 1.0
	ASM_SIZE_DIRECTIVE(one)
	.type l2e,@object
l2e:	.tfloat 1.442695040888963407359924681002
	ASM_SIZE_DIRECTIVE(l2e)

DEFINE_DBL_MIN

#ifdef PIC
#define MO(op) op##@GOTOFF(%edx)
#else
#define MO(op) op
#endif

	.text
ENTRY(__expm1)
	movzwl	4+6(%esp), %eax
	xorb	$0x80, %ah	// invert sign bit (now 1 is "positive")
	cmpl	$0xc086, %eax	// is num >= 704?
	jae	HIDDEN_JUMPTARGET (__exp)

	fldl	4(%esp)		// x
	fxam			// Is NaN, +-Inf or +-0?
	xorb	$0x80, %ah
	cmpl	$0xc043, %eax	// is num <= -38.0?
	fstsw	%ax
	movb	$0x45, %ch
	jb	4f

	// Below -38.0 (may be -NaN or -Inf).
	andb	%ah, %ch
#ifdef	PIC
	LOAD_PIC_REG (dx)
#endif
	cmpb	$0x01, %ch
	je	5f		// If -NaN, jump.
	jmp	2f		// -large, possibly -Inf.

4:	// In range -38.0 to 704.0 (may be +-0 but not NaN or +-Inf).
	andb	%ah, %ch
	cmpb	$0x40, %ch
	je	3f		// If +-0, jump.
#ifdef	PIC
	LOAD_PIC_REG (dx)
#endif

5:	fldt	MO(l2e)		// log2(e) : x
	fmulp			// log2(e)*x
	fld	%st		// log2(e)*x : log2(e)*x
	// Set round-to-nearest temporarily.
	subl	$8, %esp
	cfi_adjust_cfa_offset (8)
	fstcw	4(%esp)
	movl	$0xf3ff, %ecx
	andl	4(%esp), %ecx
	movl	%ecx, (%esp)
	fldcw	(%esp)
	frndint			// int(log2(e)*x) : log2(e)*x
	fldcw	4(%esp)
	addl	$8, %esp
	cfi_adjust_cfa_offset (-8)
	fsubr	%st, %st(1)	// int(log2(e)*x) : fract(log2(e)*x)
	fxch			// fract(log2(e)*x) : int(log2(e)*x)
	f2xm1			// 2^fract(log2(e)*x)-1 : int(log2(e)*x)
	fscale			// 2^(log2(e)*x)-2^int(log2(e)*x) : int(log2(e)*x)
	fxch			// int(log2(e)*x) : 2^(log2(e)*x)-2^int(log2(e)*x)
	fldl	MO(one)		// 1 : int(log2(e)*x) : 2^(log2(e)*x)-2^int(log2(e)*x)
	fscale			// 2^int(log2(e)*x) : int(log2(e)*x) : 2^(log2(e)*x)-2^int(log2(e)*x)
	fsubrl	MO(one)		// 1-2^int(log2(e)*x) : int(log2(e)*x) : 2^(log2(e)*x)-2^int(log2(e)*x)
	fstp	%st(1)		// 1-2^int(log2(e)*x) : 2^(log2(e)*x)-2^int(log2(e)*x)
	fsubrp	%st, %st(1)	// 2^(log2(e)*x)
	DBL_CHECK_FORCE_UFLOW
	ret

2:	fstp	%st
	fldl	MO(minus1)	// Set result to -1.0.
3:	ret
END(__expm1)
libm_alias_double (__expm1, expm1)

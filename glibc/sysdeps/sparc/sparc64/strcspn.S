/* strcspn (str, ss) -- Return the length of the initial segment of STR
                        which contains no characters from SS.
   For SPARC v9.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jj@ultra.linux.cz>

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

#include <sysdep.h>
#include <asm/asi.h>
#ifndef XCC
#define XCC xcc
#define STACK_SIZE	128
#define STACK_OFFSET	128+0x7ff
	.register	%g2, #scratch
#else
#define STACK_SIZE	64
#define STACK_OFFSET	64
#endif

	.text
	.align		32
ENTRY(strcspn)
	sub		%sp, STACK_SIZE+32, %sp		/* IEU0		Group		*/
	mov		1, %o4				/* IEU1				*/
	stx		%o4, [%sp + STACK_OFFSET]	/* Store	Group		*/
	mov		%o0, %g4			/* IEU0				*/

	stx		%g0, [%sp + STACK_OFFSET + 8]	/* Store	Group		*/
	add		%sp, STACK_OFFSET, %o5		/* IEU0				*/
	stx		%g0, [%sp + STACK_OFFSET + 16]	/* Store	Group		*/
	stx		%g0, [%sp + STACK_OFFSET + 24]	/* Store	Group		*/

1:	ldub		[%o1], %o2			/* Load		Group		*/
	brz,pn		%o2, 2f				/* CTI+IEU1	Group		*/
	 srl		%o2, 3, %o3			/* IEU0				*/
	and		%o3, 0x18, %o3			/* IEU0		Group		*/

	and		%o2, 0x3f, %o2			/* IEU1				*/
	ldx		[%o5 + %o3], %g2		/* Load		Group		*/
	sllx		%o4, %o2, %g1			/* IEU0				*/
	add		%o1, 1, %o1			/* IEU1				*/

	or		%g2, %g1, %g2			/* IEU0		Group		*/
	ba,pt		%xcc, 1b			/* CTI				*/
	 stx		%g2, [%o5 + %o3]		/* Store			*/
2:	andcc		%o0, 7, %g0			/* IEU1		Group		*/

	be,a,pt		%xcc, 4f			/* CTI				*/
	 ldx		[%o0], %o2			/* Load				*/
	ldub		[%o0], %o2			/* Load		Group		*/
3:	srl		%o2, 3, %o3			/* IEU0		Group		*/

	and		%o2, 0x3f, %o2			/* IEU1				*/
	and		%o3, 0x18, %o3			/* IEU0		Group		*/
	ldx		[%o5 + %o3], %g2		/* Load		Group		*/
	sllx		%o4, %o2, %g1			/* IEU0				*/

	add		%o0, 1, %o0			/* IEU1				*/
	andcc		%g2, %g1, %g0			/* IEU1		Group		*/
	bne,pn		%xcc, 12f			/* CTI				*/
	 andcc		%o0, 7, %g0			/* IEU1		Group		*/

	bne,a,pt	%icc, 3b			/* CTI				*/
	 ldub		[%o0], %o2			/* Load				*/
	ldx		[%o0], %o2			/* Load		Group		*/
4:	srlx		%o2, 59, %o3			/* IEU0		Group		*/

	srlx		%o2, 56, %g5			/* IEU0		Group		*/
5:	and		%o3, 0x18, %o3			/* IEU1				*/
	andcc		%g5, 0x3f, %g5			/* IEU1		Group		*/
	ldx		[%o5 + %o3], %g2		/* Load				*/

	srlx		%o2, 51, %o3			/* IEU0				*/
	sllx		%o4, %g5, %g1			/* IEU0		Group		*/
	srlx		%o2, 48, %g5			/* IEU0		Group		*/
	andcc		%g2, %g1, %g2			/* IEU1				*/

	bne,pn		%xcc, 13f			/* CTI				*/
	 and		%o3, 0x18, %o3			/* IEU0		Group		*/
	and		%g5, 0x3f, %g5			/* IEU1				*/
	ldx		[%o5 + %o3], %g2		/* Load		Group		*/

	srlx		%o2, 43, %o3			/* IEU0				*/
	sllx		%o4, %g5, %g1			/* IEU0		Group		*/
	srlx		%o2, 40, %g5			/* IEU0		Group		*/
	andcc		%g2, %g1, %g2			/* IEU1				*/

	bne,pn		%xcc, 14f			/* CTI				*/
	 and		%o3, 0x18, %o3			/* IEU0		Group		*/
	and		%g5, 0x3f, %g5			/* IEU1				*/
	ldx		[%o5 + %o3], %g2		/* Load		Group		*/

	srlx		%o2, 35, %o3			/* IEU0				*/
	sllx		%o4, %g5, %g1			/* IEU0		Group		*/
	srlx		%o2, 32, %g5			/* IEU0		Group		*/
	andcc		%g2, %g1, %g2			/* IEU1				*/

	bne,pn		%xcc, 15f			/* CTI				*/
	 and		%o3, 0x18, %o3			/* IEU0		Group		*/
	and		%g5, 0x3f, %g5			/* IEU1				*/
	ldx		[%o5 + %o3], %g2		/* Load		Group		*/

	srlx		%o2, 27, %o3			/* IEU0				*/
	sllx		%o4, %g5, %g1			/* IEU0		Group		*/
	srlx		%o2, 24, %g5			/* IEU0		Group		*/
	andcc		%g2, %g1, %g2			/* IEU1				*/

	bne,pn		%xcc, 16f			/* CTI				*/
	 and		%o3, 0x18, %o3			/* IEU0		Group		*/
	and		%g5, 0x3f, %g5			/* IEU1				*/
	ldx		[%o5 + %o3], %g2		/* Load		Group		*/

	srlx		%o2, 19, %o3			/* IEU0				*/
	sllx		%o4, %g5, %g1			/* IEU0		Group		*/
	srlx		%o2, 16, %g5			/* IEU0		Group		*/
	andcc		%g2, %g1, %g2			/* IEU1				*/

	bne,pn		%xcc, 17f			/* CTI				*/
	 and		%o3, 0x18, %o3			/* IEU0		Group		*/
	and		%g5, 0x3f, %g5			/* IEU1				*/
	ldx		[%o5 + %o3], %g2		/* Load		Group		*/

	srlx		%o2, 11, %o3			/* IEU0				*/
	sllx		%o4, %g5, %g1			/* IEU0		Group		*/
	add		%o0, 8, %o0			/* IEU1				*/
	srlx		%o2, 8, %g5			/* IEU0		Group		*/

	andcc		%g2, %g1, %g2			/* IEU1				*/
	bne,pn		%xcc, 18f			/* CTI				*/
	 and		%o3, 0x18, %o3			/* IEU0		Group		*/
	and		%g5, 0x3f, %g5			/* IEU1				*/

	ldx		[%o5 + %o3], %g2		/* Load		Group		*/
	sllx		%o4, %g5, %g1			/* IEU0				*/
	mov		%o2, %g5			/* IEU1				*/
	srlx		%o2, 3, %o3			/* IEU0		Group		*/

	ldxa		[%o0] ASI_PNF, %o2		/* Load				*/
	andcc		%g2, %g1, %g2			/* IEU1		Group		*/
	bne,pn		%xcc, 19f			/* CTI				*/
	 and		%o3, 0x18, %o3			/* IEU0		Group		*/

	and		%g5, 0x3f, %g5			/* IEU1				*/
	ldx		[%o5 + %o3], %g2		/* Load		Group		*/
	sllx		%o4, %g5, %g1			/* IEU0				*/
	srlx		%o2, 59, %o3			/* IEU0		Group		*/

	andcc		%g2, %g1, %g2			/* IEU1		Group		*/
	be,pt		%xcc, 5b			/* CTI				*/
	 srlx		%o2, 56, %g5			/* IEU0		Group		*/
	sub		%o0, 1, %o0			/* IEU1				*/

	add		%sp, STACK_SIZE+32, %sp		/* IEU0		Group		*/
	retl						/* CTI+IEU1	Group		*/
	 sub		%o0, %g4, %o0			/* IEU0				*/

	.align		16
19:	sub		%o0, 2, %o0			/* IEU1				*/
	add		%sp, STACK_SIZE+32, %sp		/* IEU0		Group		*/
	retl						/* CTI+IEU1	Group		*/
	 sub		%o0, %g4, %o0			/* IEU0				*/

18:	sub		%o0, 3, %o0			/* IEU1				*/
	add		%sp, STACK_SIZE+32, %sp		/* IEU0		Group		*/
	retl						/* CTI+IEU1	Group		*/
	 sub		%o0, %g4, %o0			/* IEU0				*/

17:	add		%o0, 4, %o0			/* IEU1				*/
	add		%sp, STACK_SIZE+32, %sp		/* IEU0		Group		*/
	retl						/* CTI+IEU1	Group		*/
	 sub		%o0, %g4, %o0			/* IEU0				*/

16:	add		%o0, 3, %o0			/* IEU1				*/
	add		%sp, STACK_SIZE+32, %sp		/* IEU0		Group		*/
	retl						/* CTI+IEU1	Group		*/
	 sub		%o0, %g4, %o0			/* IEU0				*/

15:	add		%o0, 2, %o0			/* IEU1				*/
	add		%sp, STACK_SIZE+32, %sp		/* IEU0		Group		*/
	retl						/* CTI+IEU1	Group		*/
	 sub		%o0, %g4, %o0			/* IEU0				*/

14:	add		%o0, 1, %o0			/* IEU1				*/
	add		%sp, STACK_SIZE+32, %sp		/* IEU0		Group		*/
	retl						/* CTI+IEU1	Group		*/
	 sub		%o0, %g4, %o0			/* IEU0				*/

13:	add		%sp, STACK_SIZE+32, %sp		/* IEU1				*/
	retl						/* CTI+IEU1	Group		*/
	 sub		%o0, %g4, %o0			/* IEU0				*/

	.align		16
12:	sub		%o0, 1, %o0			/* IEU0		Group		*/
	add		%sp, STACK_SIZE+32, %sp		/* IEU1				*/
	retl						/* CTI+IEU1	Group		*/
	 sub		%o0, %g4, %o0			/* IEU0				*/
END(strcspn)
libc_hidden_builtin_def (strcspn)

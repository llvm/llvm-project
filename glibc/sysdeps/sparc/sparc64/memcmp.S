/* Compare two memory blocks for differences in the first COUNT bytes.
   For SPARC v9.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jan Vondrák <jvon4518@ss1000.ms.mff.cuni.cz> and
                  Jakub Jelinek <jj@ultra.linux.cz>.

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
#define USE_BPR
	.register	%g2, #scratch
	.register	%g3, #scratch
#endif

	.text
	.align		32
ENTRY(memcmp)
#ifdef USE_BPR
	brz,pn		%o2, 3f				/* CTI+IEU1	Group		*/
#else
	tst		%o2				/* IEU1		Group		*/
	be,pn		%XCC, 3f			/* CTI				*/
#endif
	 andcc		%o0, 7, %g0			/* IEU1		Group		*/
	bne,pn		%icc, 8f			/* CTI				*/
1:	 andcc		%o1, 7, %g1			/* IEU1		Group		*/

	bne,pn		%icc, 10f			/* CTI				*/
	 mov		64, %g3				/* IEU0				*/
	ldx		[%o0], %g1			/* Load		Group		*/
	sub		%o1, %o0, %o1			/* IEU0				*/

	ldx		[%o0 + %o1], %g2		/* Load		Group		*/
	add		%o0, 8, %o0			/* IEU0				*/
2:	mov		%g1, %o3			/* IEU0		Group		*/
	subcc		%o2, 8, %o2			/* IEU1				*/

	bcs,pn		%XCC, 5f			/* CTI				*/
	 ldxa		[%o0] ASI_PNF, %g1		/* Load		Group		*/
	mov		%g2, %o4			/* IEU0				*/
	ldxa		[%o0 + %o1] ASI_PNF, %g2	/* Load		Group		*/

	cmp		%o3, %o4			/* IEU1				*/
	be,pt		%xcc, 2b			/* CTI				*/
	 add		%o0, 8, %o0			/* IEU0				*/
7:	mov		-1, %o0				/* IEU1				*/

	retl						/* CTI+IEU1	Group		*/
	 movgu		%xcc, 1, %o0			/* Single	Group		*/
3:	retl						/* CTI+IEU1	Group		*/
	 clr		%o0				/* IEU0				*/

	.align		16
5:	mov		%g2, %o4			/* IEU0				*/
6:	cmp		%o2, -8				/* IEU1				*/
	be,pn		%XCC, 3b			/* CTI				*/
	 sub		%g0, %o2, %o2			/* IEU0		Group		*/

	sllx		%o2, 3, %o2			/* IEU0		Group		*/
	srlx		%o3, %o2, %o3			/* IEU0		Group		*/
	srlx		%o4, %o2, %o4			/* IEU0		Group		*/
	clr		%o0				/* IEU1				*/

	cmp		%o3, %o4			/* IEU1		Group		*/
	movgu		%xcc, 1, %o0			/* Single	Group		*/
	retl						/* CTI+IEU1	Group		*/
	 movlu		%xcc, -1, %o0			/* Single	Group		*/

8:	ldub		[%o0], %o3			/* Load				*/
	add		%o0, 1, %o0			/* IEU0				*/
	ldub		[%o1], %o4			/* Load		Group		*/
	add		%o1, 1, %o1			/* IEU0				*/

9:	cmp		%o3, %o4			/* IEU1		Group		*/
	bne,pn		%xcc, 12f			/* CTI				*/
	 subcc		%o2, 1, %o2			/* IEU1		Group		*/
	be,pn		%XCC, 3b			/* CTI				*/

	 lduba		[%o0] ASI_PNF, %o3		/* Load				*/
	andcc		%o0, 7, %g0			/* IEU1		Group		*/
	be,pn		%icc, 1b			/* CTI				*/
	 lduba		[%o1] ASI_PNF, %o4		/* Load				*/

	add		%o0, 1, %o0			/* IEU0		Group		*/
	ba,pt		%xcc, 9b			/* CTI				*/
	 add		%o1, 1, %o1			/* IEU1				*/

	.align		16
12:	mov		-1, %o0				/* IEU0		Group		*/
	cmp		%o3, %o4			/* IEU1				*/
	retl						/* CTI+IEU1	Group		*/
	 movgu		%xcc, 1, %o0			/* Single	Group		*/

	.align		16
	nop						/* Stub				*/
10:	sllx		%g1, 3, %g2			/* IEU0		Group		*/
	sub		%o1, %g1, %o1			/* IEU1				*/
	sub		%g3, %g2, %g3			/* IEU0		Group		*/

	ldxa		[%o0] ASI_PNF, %g5		/* Load				*/
	sub		%o1, %o0, %o1			/* IEU1				*/
	ldxa		[%o0 + %o1] ASI_PNF, %g4	/* Load		Group		*/
	add		%o0, 8, %o0			/* IEU0				*/

11:	sllx		%g4, %g2, %o4			/* IEU0		Group		*/
	ldxa		[%o0 + %o1] ASI_PNF, %g4	/* Load				*/
	srlx		%g4, %g3, %o5			/* IEU0		Group		*/
	mov		%g5, %o3			/* IEU1				*/

	ldxa		[%o0] ASI_PNF, %g5		/* Load				*/
	subcc		%o2, 8, %o2			/* IEU1		Group		*/
	bcs,pn		%XCC, 6b			/* CTI				*/
	 or		%o4, %o5, %o4			/* IEU0				*/

	cmp		%o3, %o4			/* IEU1		Group		*/
	be,pt		%xcc, 11b			/* CTI				*/
	 add		%o0, 8, %o0			/* IEU0				*/
	mov		-1, %o0				/* IEU0				*/

	retl						/* CTI+IEU1	Group		*/
	 movgu		%xcc, 1, %o0			/* Single	Group		*/
END(memcmp)

#undef bcmp
weak_alias (memcmp, bcmp)
libc_hidden_builtin_def (memcmp)

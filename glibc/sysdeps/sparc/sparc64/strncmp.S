/* Compare no more than N characters of S1 and S2, returning less than,
   equal to or greater than zero if S1 is lexicographically less than,
   equal to or greater than S2.
   For SPARC v9.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
	.register	%g6, #scratch
#endif

	/* Normally, this uses
	   ((xword - 0x0101010101010101) & 0x8080808080808080) test
	   to find out if any byte in xword could be zero. This is fast, but
	   also gives false alarm for any byte in range 0x81-0xff. It does
	   not matter for correctness, as if this test tells us there could
	   be some zero byte, we check it byte by byte, but if bytes with
	   high bits set are common in the strings, then this will give poor
	   performance. You can #define EIGHTBIT_NOT_RARE and the algorithm
	   will use one tick slower, but more precise test
	   ((xword - 0x0101010101010101) & (~xword) & 0x8080808080808080),
	   which does not give any false alarms (but if some bits are set,
	   one cannot assume from it which bytes are zero and which are not).
	   It is yet to be measured, what is the correct default for glibc
	   in these days for an average user.
	 */

	.text
	.align		32
ENTRY(strncmp)
#ifdef USE_BPR
	brz,pn		%o2, 4f				/* CTI+IEU1	Group		*/
#else
	tst		%o2				/* IEU1		Group		*/
	be,pn		%XCC, 4f			/* CTI				*/
#endif
	 sethi		%hi(0x1010101), %g1		/* IEU0				*/
	andcc		%o0, 7, %g0			/* IEU1		Group		*/
	bne,pn		%icc, 9f			/* CTI				*/

	 or		%g1, %lo(0x1010101), %g1	/* IEU0				*/
	andcc		%o1, 7, %g3			/* IEU1		Group		*/
	bne,pn		%icc, 11f			/* CTI				*/
	 sllx		%g1, 32, %g2			/* IEU0				*/

	ldx		[%o0], %g4			/* Load		Group		*/
	or		%g1, %g2, %g1			/* IEU0				*/
1:	ldx		[%o1], %o3			/* Load		Group		*/
	sllx		%g1, 7, %g2			/* IEU0				*/

	add		%o0, 8, %o0			/* IEU1				*/
2:	subcc		%o2, 8, %o2			/* IEU1		Group		*/
	bcs,pn		%XCC, 5f			/* CTI				*/
	 add		%o1, 8, %o1			/* IEU0				*/

	sub		%g4, %g1, %g3			/* IEU0		Group		*/
	subcc		%g4, %o3, %o4			/* IEU1				*/
#ifdef EIGHTBIT_NOT_RARE
	andn		%g3, %g4, %g6			/* IEU0		Group		*/
#endif
	bne,pn		%xcc, 6f			/* CTI				*/
	 ldxa		[%o0] ASI_PNF, %g4		/* Load		Group		*/

	add		%o0, 8, %o0			/* IEU0				*/
#ifdef EIGHTBIT_NOT_RARE
	andcc		%g6, %g2, %g0			/* IEU1				*/
#else
	andcc		%g3, %g2, %g0			/* IEU1				*/
#endif
	be,a,pt		%xcc, 2b			/* CTI				*/
	 ldxa		[%o1] ASI_PNF, %o3		/* Load		Group		*/

	addcc		%g3, %g1, %o4			/* IEU1				*/
#ifdef EIGHTBIT_NOT_RARE
	srlx		%g6, 32, %g6			/* IEU0				*/
	andcc		%g6, %g2, %g0			/* IEU1		Group		*/
#else
	srlx		%g3, 32, %g3			/* IEU0				*/
	andcc		%g3, %g2, %g0			/* IEU1		Group		*/
#endif
	be,pt		%xcc, 3f			/* CTI				*/

	 srlx		%o4, 56, %o5			/* IEU0				*/
	andcc		%o5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 4f			/* CTI				*/
	 srlx		%o4, 48, %o5			/* IEU0				*/

	andcc		%o5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 4f			/* CTI				*/
	 srlx		%o4, 40, %o5			/* IEU0				*/
	andcc		%o5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 4f			/* CTI				*/
	 srlx		%o4, 32, %o5			/* IEU0				*/
	andcc		%o5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 4f			/* CTI				*/

3:	 srlx		%o4, 24, %o5			/* IEU0				*/
	andcc		%o5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 4f			/* CTI				*/
	 srlx		%o4, 16, %o5			/* IEU0				*/

	andcc		%o5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 4f			/* CTI				*/
	 srlx		%o4, 8, %o5			/* IEU0				*/
	andcc		%o5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 4f			/* CTI				*/
	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/
	bne,a,pn	%icc, 2b			/* CTI				*/
	 ldxa		[%o1] ASI_PNF, %o3		/* Load				*/

4:	retl						/* CTI+IEU1	Group		*/
	 clr		%o0				/* IEU0				*/

	.align		16
5:	srlx		%g4, 56, %o4			/* IEU0		Group		*/
	cmp		%o2, -8				/* IEU1				*/
	be,pn		%XCC, 4b			/* CTI				*/
	 srlx		%o3, 56, %o5			/* IEU0		Group		*/

	andcc		%o4, 0xff, %g0			/* IEU1				*/
	be,pn		%xcc, 8f			/* CTI				*/
	 subcc		%o4, %o5, %o4			/* IEU1		Group		*/
	bne,pn		%xcc, 8f			/* CTI				*/

	 srlx		%o3, 48, %o5			/* IEU0				*/
	cmp		%o2, -7				/* IEU1		Group		*/
	be,pn		%XCC, 4b			/* CTI				*/
	 srlx		%g4, 48, %o4			/* IEU0				*/

	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%xcc, 8f			/* CTI				*/
	 subcc		%o4, %o5, %o4			/* IEU1		Group		*/
	bne,pn		%xcc, 8f			/* CTI				*/

	 srlx		%o3, 40, %o5			/* IEU0				*/
	cmp		%o2, -6				/* IEU1		Group		*/
	be,pn		%XCC, 4b			/* CTI				*/
	 srlx		%g4, 40, %o4			/* IEU0				*/

	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%xcc, 8f			/* CTI				*/
	 subcc		%o4, %o5, %o4			/* IEU1		Group		*/
	bne,pn		%xcc, 8f			/* CTI				*/

	 srlx		%o3, 32, %o5			/* IEU0				*/
	cmp		%o2, -5				/* IEU1		Group		*/
	be,pn		%XCC, 4b			/* CTI				*/
	 srlx		%g4, 32, %o4			/* IEU0				*/

	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%xcc, 8f			/* CTI				*/
	 subcc		%o4, %o5, %o4			/* IEU1		Group		*/
	bne,pn		%xcc, 8f			/* CTI				*/

	 srlx		%o3, 24, %o5			/* IEU0				*/
	cmp		%o2, -4				/* IEU1		Group		*/
	be,pn		%XCC, 4b			/* CTI				*/
	 srlx		%g4, 24, %o4			/* IEU0				*/

	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%xcc, 8f			/* CTI				*/
	 subcc		%o4, %o5, %o4			/* IEU1		Group		*/
	bne,pn		%xcc, 8f			/* CTI				*/

	 srlx		%o3, 16, %o5			/* IEU0				*/
	cmp		%o2, -3				/* IEU1		Group		*/
	be,pn		%XCC, 4b			/* CTI				*/
	 srlx		%g4, 16, %o4			/* IEU0				*/

	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%xcc, 8f			/* CTI				*/
	 subcc		%o4, %o5, %o4			/* IEU1		Group		*/
	bne,pn		%xcc, 8f			/* CTI				*/

	 srlx		%o3, 8, %o5			/* IEU0				*/
	cmp		%o2, -2				/* IEU1		Group		*/
	be,pn		%XCC, 4b			/* CTI				*/
	 srlx		%g4, 8, %o4			/* IEU0				*/

	retl						/* CTI+IEU1	Group		*/
	 sub		%o4, %o5, %o0			/* IEU0				*/
6:	addcc		%o3, %o4, %g4			/* IEU1				*/
7:	srlx		%o3, 56, %o5			/* IEU0				*/

	srlx		%g4, 56, %o4			/* IEU0		Group		*/
	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%xcc, 8f			/* CTI				*/
	 subcc		%o4, %o5, %o4			/* IEU1		Group		*/

	bne,pn		%xcc, 8f			/* CTI				*/
	 srlx		%o3, 48, %o5			/* IEU0				*/
	srlx		%g4, 48, %o4			/* IEU0		Group		*/
	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%xcc, 8f			/* CTI				*/
	 subcc		%o4, %o5, %o4			/* IEU1		Group		*/
	bne,pn		%xcc, 8f			/* CTI				*/
	 srlx		%o3, 40, %o5			/* IEU0				*/

	srlx		%g4, 40, %o4			/* IEU0		Group		*/
	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%xcc, 8f			/* CTI				*/
	 subcc		%o4, %o5, %o4			/* IEU1		Group		*/

	bne,pn		%xcc, 8f			/* CTI				*/
	 srlx		%o3, 32, %o5			/* IEU0				*/
	srlx		%g4, 32, %o4			/* IEU0		Group		*/
	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%xcc, 8f			/* CTI				*/
	 subcc		%o4, %o5, %o4			/* IEU1		Group		*/
	bne,pn		%xcc, 8f			/* CTI				*/
	 srlx		%o3, 24, %o5			/* IEU0				*/

	srlx		%g4, 24, %o4			/* IEU0		Group		*/
	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%xcc, 8f			/* CTI				*/
	 subcc		%o4, %o5, %o4			/* IEU1		Group		*/

	bne,pn		%xcc, 8f			/* CTI				*/
	 srlx		%o3, 16, %o5			/* IEU0				*/
	srlx		%g4, 16, %o4			/* IEU0		Group		*/
	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%xcc, 8f			/* CTI				*/
	 subcc		%o4, %o5, %o4			/* IEU1		Group		*/
	bne,pn		%xcc, 8f			/* CTI				*/
	 srlx		%o3, 8, %o5			/* IEU0				*/

	srlx		%g4, 8, %o4			/* IEU0		Group		*/
	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%xcc, 8f			/* CTI				*/
	 subcc		%o4, %o5, %o4			/* IEU1		Group		*/

	retl						/* CTI+IEU1	Group		*/
	 sub		%g4, %o3, %o0			/* IEU0				*/
8:	retl						/* CTI+IEU1	Group		*/
	 mov		%o4, %o0			/* IEU0				*/

9:	ldub		[%o0], %g4			/* Load		Group		*/
	add		%o0, 1, %o0			/* IEU0				*/
	ldub		[%o1], %o3			/* Load		Group		*/
	sllx		%g1, 32, %g2			/* IEU0				*/

10:	subcc		%o2, 1, %o2			/* IEU1		Group		*/
	be,pn		%XCC, 8b			/* CTI				*/
	 sub		%g4, %o3, %o4			/* IEU0				*/
	add		%o1, 1, %o1			/* IEU0		Group		*/

	cmp		%g4, %o3			/* IEU1				*/
	bne,pn		%xcc, 8b			/* CTI				*/
	 lduba		[%o0] ASI_PNF, %g4		/* Load		Group		*/
	andcc		%o3, 0xff, %g0			/* IEU1				*/

	be,pn		%icc, 4b			/* CTI				*/
	 lduba		[%o1] ASI_PNF, %o3		/* Load		Group		*/
	andcc		%o0, 7, %g0			/* IEU1				*/
	bne,a,pn	%icc, 10b			/* CTI				*/

	 add		%o0, 1, %o0			/* IEU0		Group		*/
	or		%g1, %g2, %g1			/* IEU1				*/
	andcc		%o1, 7, %g3			/* IEU1		Group		*/
	be,pn		%icc, 1b			/* CTI				*/

	 ldxa		[%o0] ASI_PNF, %g4		/* Load				*/
11:	sllx		%g3, 3, %g5			/* IEU0		Group		*/
	mov		64, %g6				/* IEU1				*/
	or		%g1, %g2, %g1			/* IEU0		Group		*/
	sub		%o1, %g3, %o1			/* IEU1				*/

	sub		%g6, %g5, %g6			/* IEU0		Group		*/
	ldxa		[%o1] ASI_PNF, %o4		/* Load				*/
	sllx		%g1, 7, %g2			/* IEU1				*/
	add		%o1, 8, %o1			/* IEU0		Group		*/
							/* %g1 = 0101010101010101
							   %g2 = 8080808080808080
							   %g3 = %o1 alignment
							   %g5 = number of bits to shift left
							   %g6 = number of bits to shift right */

12:	sllx		%o4, %g5, %o3			/* IEU0		Group		*/
	ldxa		[%o1] ASI_PNF, %o4		/* Load				*/
	add		%o1, 8, %o1			/* IEU1				*/
13:	ldxa		[%o0] ASI_PNF, %g4		/* Load		Group		*/

	addcc		%o0, 8, %o0			/* IEU1				*/
	srlx		%o4, %g6, %o5			/* IEU0				*/
	subcc		%o2, 8, %o2			/* IEU1		Group		*/
	bcs,pn		%XCC, 5b			/* CTI				*/

	 or		%o3, %o5, %o3			/* IEU0				*/
	cmp		%g4, %o3			/* IEU1		Group		*/
	bne,pn		%xcc, 7b			/* CTI				*/
	 sub		%g4, %g1, %o5			/* IEU0				*/

#ifdef EIGHTBIT_NOT_RARE
	andn		%o5, %g4, %o5			/* IEU0		Group		*/
#endif
	andcc		%o5, %g2, %g0			/* IEU1		Group		*/
	be,pt		%xcc, 12b			/* CTI				*/
	 srlx		%o5, 32, %o5			/* IEU0				*/
	andcc		%o5, %g2, %g0			/* IEU1		Group		*/

	be,pt		%xcc, 14f			/* CTI				*/
	 srlx		%g4, 56, %o5			/* IEU0				*/
	andcc		%o5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 4b			/* CTI				*/

	 srlx		%g4, 48, %o5			/* IEU0				*/
	andcc		%o5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 4b			/* CTI				*/
	 srlx		%g4, 40, %o5			/* IEU0				*/

	andcc		%o5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 4b			/* CTI				*/
	 srlx		%g4, 32, %o5			/* IEU0				*/
	andcc		%o5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 4b			/* CTI				*/
14:	 srlx		%g4, 24, %o5			/* IEU0				*/
	andcc		%o5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 4b			/* CTI				*/

	 srlx		%g4, 16, %o5			/* IEU0				*/
	andcc		%o5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 4b			/* CTI				*/
	 srlx		%g4, 8, %o5			/* IEU0				*/

	andcc		%o5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 4b			/* CTI				*/
	 andcc		%g4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 4b			/* CTI				*/

	 sllx		%o4, %g5, %o3			/* IEU0				*/
	ldxa		[%o1] ASI_PNF, %o4		/* Load		Group		*/
	ba,pt		%xcc, 13b			/* CTI				*/
	 add		%o1, 8, %o1			/* IEU0				*/
END(strncmp)
libc_hidden_builtin_def (strncmp)

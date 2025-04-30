/* strchr (str, ch) -- Return pointer to first occurrence of CH in STR.
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
ENTRY(strchr)
	andcc		%o1, 0xff, %o1			/* IEU1		Group		*/
	be,pn		%icc, 17f			/* CTI				*/
	 sllx		%o1, 8, %g3			/* IEU0		Group		*/
	sethi		%hi(0x01010101), %g1		/* IEU1				*/

	or		%g3, %o1, %g3			/* IEU0		Group		*/
	ldub		[%o0], %o3			/* Load				*/
	sllx		%g3, 16, %g5			/* IEU0		Group		*/
	or		%g1, %lo(0x01010101), %g1	/* IEU1				*/

	sllx		%g1, 32, %g2			/* IEU0		Group		*/
	brz,pn		%o3, 5f				/* CTI+IEU1			*/
	 orcc		%g3, %g5, %g3			/* IEU1		Group		*/
	sllx		%g3, 32, %g5			/* IEU0				*/

	cmp		%o3, %o1			/* IEU1		Group		*/
	be,pn		%xcc, 14f			/* CTI				*/
	 or		%g1, %g2, %g1			/* IEU0				*/
	andcc		%o0, 7, %g0			/* IEU1		Group		*/

	bne,a,pn	%icc, 15f			/* CTI				*/
	 add		%o0, 1, %o0			/* IEU0				*/
	ldx		[%o0], %o3			/* Load		Group		*/
1:	sllx		%g1, 7, %g2			/* IEU0				*/

	or		%g3, %g5, %g3			/* IEU1				*/
	add		%o0, 8, %o0			/* IEU0		Group		*/
	xor		%o3, %g3, %o4			/* IEU1				*/
							/* %g1 = 0101010101010101	*
							 * %g2 = 8080088080808080	*
							 * %g3 =  c c c c c c c c	*
							 * %o3 =      value		*
							 * %o4 =   value XOR c		*/
2:	sub		%o3, %g1, %o2			/* IEU0		Group		*/

	sub		%o4, %g1, %o5			/* IEU1				*/
#ifdef EIGHTBIT_NOT_RARE
	andn		%o2, %o3, %g6			/* IEU0		Group		*/
	andn		%o5, %o4, %o5			/* IEU1				*/
	ldxa		[%o0] ASI_PNF, %o3		/* Load				*/
	or		%o5, %g6, %o5			/* IEU0		Group		*/
#else
	ldxa		[%o0] ASI_PNF, %o3		/* Load				*/
	or		%o5, %o2, %o5			/* IEU0		Group		*/
#endif
	add		%o0, 8, %o0			/* IEU1				*/

	andcc		%o5, %g2, %g0			/* IEU1		Group		*/
	be,a,pt		%xcc, 2b			/* CTI				*/
	 xor		%o3, %g3, %o4			/* IEU0				*/
	srlx		%o5, 32, %g5			/* IEU0		Group		*/

	add		%o2, %g1, %o2			/* IEU1				*/
3:	andcc		%g5, %g2, %g0			/* IEU1		Group		*/
	be,pn		%xcc, 4f			/* CTI				*/
	 srlx		%o2, 56, %g5			/* IEU0				*/

	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 5f			/* CTI				*/
	 srlx		%o4, 56, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 6f			/* CTI				*/
	 srlx		%o2, 48, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 5f			/* CTI				*/

	 srlx		%o4, 48, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 7f			/* CTI				*/
	 srlx		%o2, 40, %g5			/* IEU0				*/

	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 5f			/* CTI				*/
	 srlx		%o4, 40, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 8f			/* CTI				*/
	 srlx		%o2, 32, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 5f			/* CTI				*/

	 srlx		%o4, 32, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 9f			/* CTI				*/
4:	 srlx		%o2, 24, %g5			/* IEU0				*/

	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 5f			/* CTI				*/
	 srlx		%o4, 24, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 10f			/* CTI				*/
	 srlx		%o2, 16, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 5f			/* CTI				*/

	 srlx		%o4, 16, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 11f			/* CTI				*/
	 srlx		%o2, 8, %g5			/* IEU0				*/

	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 5f			/* CTI				*/
	 srlx		%o4, 8, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 12f			/* CTI				*/
	andcc		%o2, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 5f			/* CTI				*/
	 sub		%o3, %g1, %o2			/* IEU0				*/

	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 13f			/* CTI				*/
	 xor		%o3, %g3, %o4			/* IEU0				*/
	ldxa		[%o0] ASI_PNF, %o3		/* Load		Group		*/

	sub		%o4, %g1, %o5			/* IEU0				*/
	or		%o5, %o2, %o5			/* IEU1				*/
	add		%o0, 8, %o0			/* IEU0		Group		*/
	andcc		%o5, %g2, %g0			/* IEU1				*/

	be,a,pt		%xcc, 2b			/* CTI				*/
	 xor		%o3, %g3, %o4			/* IEU0		Group		*/
	srlx		%o5, 32, %g5			/* IEU0		Group		*/
	ba,pt		%xcc, 3b			/* CTI				*/

	 add		%o2, %g1, %o2			/* IEU1				*/

	.align		16
5:	retl						/* CTI+IEU1	Group		*/
	 clr		%o0				/* IEU0				*/
6:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -16, %o0			/* IEU0				*/

7:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -15, %o0			/* IEU0				*/
8:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -14, %o0			/* IEU0				*/

9:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -13, %o0			/* IEU0				*/
10:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -12, %o0			/* IEU0				*/

11:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -11, %o0			/* IEU0				*/
12:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -10, %o0			/* IEU0				*/

13:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -9, %o0			/* IEU0				*/
14: 	retl						/* CTI+IEU1	Group		*/
	 nop						/* IEU0				*/

	.align		16
15:	ldub		[%o0], %o3			/* Load		Group		*/
16:	andcc		%o0, 7, %g0			/* IEU1				*/
	be,a,pn		%icc, 1b			/* CTI				*/
	 ldx		[%o0], %o3			/* Load		Group		*/

	andcc		%o3, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 5b			/* CTI				*/
	 add		%o0, 1, %o0			/* IEU0				*/
	cmp		%o3, %o1			/* IEU1		Group		*/

	bne,a,pn	%icc, 16b			/* CTI				*/
	 ldub		[%o0], %o3			/* Load				*/
	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -1, %o0			/* IEU0				*/

	/* strchr (str, 0)			*/
	.align		32
	nop
	.align		16
17:	sethi		%hi(0x01010101), %g1		/* IEU0		Group		*/
	ldub		[%o0], %o3			/* Load				*/
	or		%g1, %lo(0x01010101), %g1	/* IEU0		Group		*/
	sllx		%g1, 32, %g2			/* IEU0		Group		*/

	andcc		%o0, 7, %g0			/* IEU1				*/
	or		%g1, %g2, %g1			/* IEU0		Group		*/
	bne,pn		%icc, 32f			/* CTI				*/
	 sllx		%g1, 7, %g2			/* IEU0		Group		*/

	brz,pn		%o3, 30f			/* CTI+IEU1			*/
	 ldx		[%o0], %o3			/* Load				*/
18:	add		%o0, 8, %o0			/* IEU0		Group		*/
19:	sub		%o3, %g1, %o2			/* IEU0		Group		*/

#ifdef EIGHTBIT_NOT_RARE
	andn		%o2, %o3, %g6			/* IEU0		Group		*/
	ldxa		[%o0] ASI_PNF, %o3		/* Load				*/
	andcc		%g6, %g2, %g0			/* IEU1		Group		*/
#else
	ldxa		[%o0] ASI_PNF, %o3		/* Load				*/
	andcc		%o2, %g2, %g0			/* IEU1		Group		*/
#endif
	be,pt		%xcc, 19b			/* CTI				*/
	 add		%o0, 8, %o0			/* IEU0				*/

 	addcc		%o2, %g1, %g3			/* IEU1		Group		*/
	srlx		%o2, 32, %o2			/* IEU0				*/
20:	andcc		%o2, %g2, %g0			/* IEU1		Group		*/
	be,pn		%xcc, 21f			/* CTI				*/

	 srlx		%g3, 56, %o2			/* IEU0				*/
	andcc		%o2, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 29f			/* CTI				*/
	 srlx		%g3, 48, %o2			/* IEU0				*/

	andcc		%o2, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 28f			/* CTI				*/
	 srlx		%g3, 40, %o2			/* IEU0				*/
	andcc		%o2, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 27f			/* CTI				*/
	 srlx		%g3, 32, %o2			/* IEU0				*/
	andcc		%o2, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 26f			/* CTI				*/

21:	 srlx		%g3, 24, %o2			/* IEU0				*/
	andcc		%o2, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 25f			/* CTI				*/
	 srlx		%g3, 16, %o2			/* IEU0				*/

	andcc		%o2, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 24f			/* CTI				*/
	 srlx		%g3, 8, %o2			/* IEU0				*/
	andcc		%o2, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 23f			/* CTI				*/
	 sub		%o3, %g1, %o2			/* IEU0				*/
	andcc		%g3, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 22f			/* CTI				*/

	 ldxa		[%o0] ASI_PNF, %o3		/* Load				*/
	andcc		%o2, %g2, %g0			/* IEU1		Group		*/
	be,pt		%xcc, 19b			/* CTI				*/
	 add		%o0, 8, %o0			/* IEU0				*/

	addcc		%o2, %g1, %g3			/* IEU1		Group		*/
	ba,pt		%xcc, 20b			/* CTI				*/
	 srlx		%o2, 32, %o2			/* IEU0				*/

	.align		16
22:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -9, %o0			/* IEU0				*/
23:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -10, %o0			/* IEU0				*/

24:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -11, %o0			/* IEU0				*/
25:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -12, %o0			/* IEU0				*/

26:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -13, %o0			/* IEU0				*/
27:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -14, %o0			/* IEU0				*/

28:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -15, %o0			/* IEU0				*/
29:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -16, %o0			/* IEU0				*/

30:	retl						/* CTI+IEU1	Group		*/
	 nop						/* IEU0				*/

	.align		16
32:	andcc		%o0, 7, %g0			/* IEU1		Group		*/
	be,a,pn		%icc, 18b			/* CTI				*/
	 ldx		[%o0], %o3			/* Load				*/
	add		%o0, 1, %o0			/* IEU0		Group		*/

	brnz,a,pt	%o3, 32b			/* CTI+IEU1			*/
	 lduba		[%o0] ASI_PNF, %o3		/* Load				*/
	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -1, %o0			/* IEU0				*/
END(strchr)

	.align		32
ENTRY(strrchr)
	andcc		%o1, 0xff, %o1			/* IEU1		Group		*/
	be,pn		%icc, 17b			/* CTI				*/
	 clr		%g4				/* IEU0				*/
	andcc		%o0, 7, %g0			/* IEU1		Group		*/

	bne,pn		%icc, 13f			/* CTI				*/
	 sllx		%o1, 8, %g3			/* IEU0				*/
	ldx		[%o0], %o3			/* Load		Group		*/
1:	sethi		%hi(0x01010101), %g1		/* IEU0				*/

	or		%g3, %o1, %g3			/* IEU1				*/
	sllx		%g3, 16, %g5			/* IEU0		Group		*/
	or		%g1, %lo(0x01010101), %g1	/* IEU1				*/
	sllx		%g1, 32, %g2			/* IEU0		Group		*/

	or		%g3, %g5, %g3			/* IEU1				*/
	sllx		%g3, 32, %g5			/* IEU0		Group		*/
	or		%g1, %g2, %g1			/* IEU1				*/
	sllx		%g1, 7, %g2			/* IEU0		Group		*/

	or		%g3, %g5, %g3			/* IEU1				*/
	add		%o0, 8, %o0			/* IEU0		Group		*/
	xor		%o3, %g3, %o4			/* IEU1				*/
							/* %g1 = 0101010101010101	*
							 * %g2 = 8080088080808080	*
							 * %g3 =  c c c c c c c c	*
							 * %o3 =   value		*
							 * %o4 =   value XOR c		*/
2:	sub		%o3, %g1, %o2			/* IEU0		Group		*/

3:	sub		%o4, %g1, %o5			/* IEU1				*/
#ifdef EIGHTBIT_NOT_RARE
	andn		%o2, %o3, %g6			/* IEU0		Group		*/
	andn		%o5, %o4, %o5			/* IEU1				*/
	ldxa		[%o0] ASI_PNF, %o3		/* Load				*/

	or		%o5, %g6, %o5			/* IEU0		Group		*/
#else
	ldxa		[%o0] ASI_PNF, %o3		/* Load				*/

	or		%o5, %o2, %o5			/* IEU0		Group		*/
#endif
	add		%o0, 8, %o0			/* IEU1				*/
	andcc		%o5, %g2, %g0			/* IEU1		Group		*/
	be,a,pt		%xcc, 2b			/* CTI				*/

	 xor		%o3, %g3, %o4			/* IEU0				*/
	srlx		%o5, 32, %g5			/* IEU0		Group		*/
	add		%o2, %g1, %o2			/* IEU1				*/
	andcc		%g5, %g2, %g0			/* IEU1		Group		*/

	be,pn		%xcc, 7f			/* CTI				*/
	 srlx		%o2, 56, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 12f			/* CTI				*/

	 srlx		%o4, 56, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	srlx		%o2, 48, %g5			/* IEU0				*/
	be,a,pn		%icc, 4f			/* CTI				*/

	 add		%o0, -16, %g4			/* IEU0		Group		*/
4:	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 12f			/* CTI				*/
	 srlx		%o4, 48, %g5			/* IEU0				*/

	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	srlx		%o2, 40, %g5			/* IEU0				*/
	be,a,pn		%icc, 5f			/* CTI				*/
	 add		%o0, -15, %g4			/* IEU0		Group		*/

5:	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 12f			/* CTI				*/
	 srlx		%o4, 40, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/

	srlx		%o2, 32, %g5			/* IEU0				*/
	be,a,pn		%icc, 6f			/* CTI				*/
	 add		%o0, -14, %g4			/* IEU0		Group		*/
6:	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 12f			/* CTI				*/
	 srlx		%o4, 32, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,a,pn		%icc, 7f			/* CTI				*/

	 add		%o0, -13, %g4			/* IEU0				*/
7:	srlx		%o2, 24, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 12f			/* CTI				*/

	 srlx		%o4, 24, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	srlx		%o2, 16, %g5			/* IEU0				*/
	be,a,pn		%icc, 8f			/* CTI				*/

	 add		%o0, -12, %g4			/* IEU0		Group		*/
8:	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 12f			/* CTI				*/
	 srlx		%o4, 16, %g5			/* IEU0				*/

	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	srlx		%o2, 8, %g5			/* IEU0				*/
	be,a,pn		%icc, 9f			/* CTI				*/
	 add		%o0, -11, %g4			/* IEU0		Group		*/

9:	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 12f			/* CTI				*/
	 srlx		%o4, 8, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/

	be,a,pn		%icc, 10f			/* CTI				*/
	 add		%o0, -10, %g4			/* IEU0				*/
10:	andcc		%o2, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 12f			/* CTI				*/

	 sub		%o3, %g1, %o2			/* IEU0				*/
	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/
	be,a,pn		%icc, 11f			/* CTI				*/
	 add		%o0, -9, %g4			/* IEU0				*/

11:	ba,pt		%xcc, 3b			/* CTI		Group		*/
	 xor		%o3, %g3, %o4			/* IEU0		Group		*/
12:	retl						/* CTI+IEU1	Group		*/
	 mov		%g4, %o0			/* IEU0				*/

	.align		16
13:	ldub		[%o0], %o3			/* Load		Group		*/
	add		%o0, 1, %o0			/* IEU0				*/
14:	andcc		%o3, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 12b			/* CTI				*/

	cmp		%o3, %o1			/* IEU1		Group		*/
	ldub		[%o0], %o3			/* Load				*/
	be,a,pn 	%icc, 15f			/* CTI				*/
	 add		%o0, -1, %g4			/* IEU0		Group		*/

15:	andcc		%o0, 7, %g0			/* IEU1		Group		*/
	bne,a,pt	%icc, 14b			/* CTI				*/
	 add		%o0, 1, %o0			/* IEU0				*/
	ba,pt		%xcc, 1b			/* CTI		Group		*/

	 ldx		[%o0], %o3			/* Load				*/
END(strrchr)

weak_alias (strchr, index)
weak_alias (strrchr, rindex)
libc_hidden_builtin_def (strchr)
libc_hidden_builtin_def (strrchr)

/* stpncpy(DST, SRC, COUNT) - Copy no more than N characters of
   SRC to DEST, returning the address of the terminating '\0' in
   DEST, if any, or else DEST + N.
   For SPARC v9.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jj@ultra.linux.cz> and
		  Jan Vondrák <jvon4518@ss1000.ms.mff.cuni.cz>.

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
ENTRY(__stpncpy)
	sethi		%hi(0x01010101), %g1		/* IEU0		Group		*/
#ifdef USE_BPR
	brz,pn		%o2, 19f			/* CTI+IEU1			*/
#else
	tst		%o2				/* IEU1				*/
	be,pn		%XCC, 19f			/* CTI				*/
#endif
	 or		%g1, %lo(0x01010101), %g1	/* IEU1				*/
	andcc		%o0, 7, %g0			/* IEU1		Group		*/

	sllx		%g1, 32, %g2			/* IEU0				*/
	bne,pn		%icc, 26f			/* CTI				*/
	 or		%g1, %g2, %g1			/* IEU0		Group		*/
	andcc		%o1, 7, %g3			/* IEU1				*/

	bne,pn		%icc, 28f			/* CTI				*/
	 sllx		%g1, 7, %g2			/* IEU0		Group		*/
	ldx		[%o1], %o3			/* Load				*/
1:	add		%o1, 8, %o1			/* IEU1				*/

2:	subcc		%o2, 8, %o2			/* IEU1		Group		*/
	bl,pn		%XCC, 18f			/* CTI				*/
	 sub		%o3, %g1, %o4			/* IEU0				*/
	add		%o0, 8, %o0			/* IEU0		Group		*/

#ifdef EIGHTBIT_NOT_MORE
	andn		%o4, %o3, %o4			/* IEU1				*/
#endif
	mov		%o3, %g3			/* IEU1				*/
	ldxa		[%o1] ASI_PNF, %o3		/* Load				*/
	add		%o1, 8, %o1			/* IEU0		Group		*/
	andcc		%o4, %g2, %g0			/* IEU1				*/

	be,a,pt		%xcc, 2b			/* CTI				*/
	 stx		%g3, [%o0-8]			/* Store	Group		*/
	srlx		%g3, 56, %g5			/* IEU0		Group		*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 16f			/* CTI				*/
	 srlx		%g3, 48, %g4			/* IEU0				*/
	andcc		%g4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 15f			/* CTI				*/

	 srlx		%g3, 40, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 14f			/* CTI				*/
	 srlx		%g3, 32, %g4			/* IEU0				*/

	andcc		%g4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 13f			/* CTI				*/
	 srlx		%g3, 24, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 12f			/* CTI				*/
	 srlx		%g3, 16, %g4			/* IEU0				*/
	andcc		%g4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 11f			/* CTI				*/

	 srlx		%g3, 8, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 10f			/* CTI				*/
	 sub		%o0, 1, %g6			/* IEU0				*/

	andcc		%g3, 0xff, %g0			/* IEU1		Group		*/
	bne,pt		%icc, 2b			/* CTI				*/
3:	 stx		%g3, [%o0-8]			/* Store			*/
	andncc		%o2, 31, %g3			/* IEU1		Group		*/

4:	be,pn		%XCC, 41f			/* CTI				*/
	 and		%o2, 31, %o2			/* IEU1		Group		*/
40:	stx		%g0, [%o0]			/* Store			*/
	stx		%g0, [%o0 + 8]			/* Store	Group		*/

	subcc		%g3, 32, %g3			/* IEU1				*/
	stx		%g0, [%o0 + 16]			/* Store	Group		*/
	stx		%g0, [%o0 + 24]			/* Store	Group		*/
	bne,pt		%XCC, 40b			/* CTI				*/

	 add		%o0, 32, %o0			/* IEU0				*/
41:	subcc		%o2, 8, %o2			/* IEU1		Group		*/
	bl,a,pn		%XCC, 6f			/* CTI				*/
	 andcc		%o2, 4, %g0			/* IEU1		Group		*/

5:	stx		%g0, [%o0]			/* Store			*/
	subcc		%o2, 8, %o2			/* IEU1		Group		*/
	bge,pt		%XCC, 5b			/* CTI				*/
	 add		%o0, 8, %o0			/* IEU0				*/

	andcc		%o2, 4, %g0			/* IEU1		Group		*/
6:	be,a,pn		%icc, 7f			/* CTI				*/
	 andcc		%o2, 2, %g0			/* IEU1		Group		*/
	stw		%g0, [%o0]			/* Store			*/

	add		%o0, 4, %o0			/* IEU0				*/
	andcc		%o2, 2, %g0			/* IEU1		Group		*/
7:	be,a,pn		%icc, 8f			/* CTI				*/
	 andcc		%o2, 1, %g0			/* IEU1		Group		*/

	sth		%g0, [%o0]			/* Store			*/
	add		%o0, 2, %o0			/* IEU0				*/
	andcc		%o2, 1, %g0			/* IEU1		Group		*/
8:	bne,a,pn	%icc, 9f			/* CTI				*/

	 stb		%g0, [%o0]			/* Store			*/
9:	retl						/* CTI+IEU1	Group		*/
	 mov		%g6, %o0			/* IEU0				*/
10:	subcc		%o0, 2, %g6			/* IEU1		Group		*/

	ba,pt		%xcc, 3b			/* CTI				*/
	 sllx		%g5, 8, %g3			/* IEU0				*/
11:	subcc		%o0, 3, %g6			/* IEU1		Group		*/
	ba,pt		%xcc, 3b			/* CTI				*/

	 sllx		%g4, 16, %g3			/* IEU0				*/
12:	subcc		%o0, 4, %g6			/* IEU1		Group		*/
	ba,pt		%xcc, 3b			/* CTI				*/
	 sllx		%g5, 24, %g3			/* IEU0				*/

13:	subcc		%o0, 5, %g6			/* IEU1		Group		*/
	ba,pt		%xcc, 3b			/* CTI				*/
	 sllx		%g4, 32, %g3			/* IEU0				*/
14:	subcc		%o0, 6, %g6			/* IEU1		Group		*/

	ba,pt		%xcc, 3b			/* CTI				*/
	 sllx		%g5, 40, %g3			/* IEU0				*/
15:	subcc		%o0, 7, %g6			/* IEU1		Group		*/
	ba,pt		%xcc, 3b			/* CTI				*/

	 sllx		%g4, 48, %g3			/* IEU0				*/
16:	subcc		%o0, 8, %g6			/* IEU1		Group		*/
	ba,pt		%xcc, 3b			/* CTI				*/
	 clr		%g3				/* IEU0				*/

	.align		16
17:	or		%o3, %o4, %o3			/* IEU0		Group		*/
	sub		%o3, %g1, %o4			/* IEU1				*/
18:	addcc		%o2, 8, %o2			/* IEU1		Group		*/
	be,pn		%XCC, 19f			/* CTI				*/

	 andcc		%o4, %g2, %g0			/* IEU1		Group		*/
	be,pt		%xcc, 21f			/* CTI				*/
	 srlx		%o3, 56, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 20f			/* CTI				*/
	 stb		%g5, [%o0]			/* Store			*/
	add		%o0, 1, %o0			/* IEU0		Group		*/
	subcc		%o2, 1, %o2			/* IEU1				*/

	be,pn		%XCC, 19f			/* CTI				*/
	 srlx		%o3, 48, %g5			/* IEU0		Group		*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 20f			/* CTI				*/

	 stb		%g5, [%o0]			/* Store			*/
	add		%o0, 1, %o0			/* IEU0		Group		*/
	subcc		%o2, 1, %o2			/* IEU1				*/
	be,pn		%XCC, 19f			/* CTI				*/

	 srlx		%o3, 40, %g5			/* IEU0		Group		*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 20f			/* CTI				*/
	 stb		%g5, [%o0]			/* Store			*/

	add		%o0, 1, %o0			/* IEU0		Group		*/
	subcc		%o2, 1, %o2			/* IEU1				*/
	be,pn		%XCC, 19f			/* CTI				*/
	 srlx		%o3, 32, %g5			/* IEU0		Group		*/

	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 20f			/* CTI				*/
	 stb		%g5, [%o0]			/* Store			*/
	add		%o0, 1, %o0			/* IEU0		Group		*/

	subcc		%o2, 1, %o2			/* IEU1				*/
	be,pn		%XCC, 19f			/* CTI				*/
	 srlx		%o3, 24, %g5			/* IEU0		Group		*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 20f			/* CTI				*/
	 stb		%g5, [%o0]			/* Store			*/
	add		%o0, 1, %o0			/* IEU0		Group		*/
	subcc		%o2, 1, %o2			/* IEU1				*/

	be,pn		%XCC, 19f			/* CTI				*/
	 srlx		%o3, 16, %g5			/* IEU0		Group		*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 20f			/* CTI				*/

	 stb		%g5, [%o0]			/* Store			*/
	add		%o0, 1, %o0			/* IEU0		Group		*/
	subcc		%o2, 1, %o2			/* IEU1				*/
	be,pn		%XCC, 19f			/* CTI				*/

	 srlx		%o3, 8, %o3			/* IEU0		Group		*/
	stb		%o3, [%o0]			/* Store			*/
59:	add		%o0, 1, %o2			/* IEU1				*/
	andcc		%o3, 0xff, %g0			/* IEU1		Group		*/

	retl						/* CTI+IEU1	Group		*/
	 movne		%icc, %o2, %o0			/* Single	Group		*/
19:	retl						/* CTI+IEU1	Group		*/
	 nop						/* IEU0				*/

20:	mov		%o0, %g6			/* IEU0		Group		*/
	subcc		%o2, 1, %o2			/* IEU1				*/
	be,pn		%XCC, 51f			/* CTI				*/
	 add		%o0, 1, %o0			/* IEU0		Group		*/

50:	stb		%g0, [%o0]			/* Store	Group		*/
	subcc		%o2, 1, %o2			/* IEU1		Group		*/
	bne,pt		%XCC, 50b			/* CTI				*/
	 add		%o0, 1, %o0			/* IEU0				*/

51:	retl						/* CTI+IEU1	Group		*/
	 mov		%g6, %o0			/* IEU0				*/

	.align		16
21:	andcc		%o2, 4, %g0			/* IEU1		Group		*/
	be,pn		%icc, 22f			/* CTI				*/
	 srlx		%o3, 32, %g5			/* IEU0				*/
	stw		%g5, [%o0]			/* Store	Group		*/

	add		%o0, 4, %o0			/* IEU0				*/
	mov		%o3, %g5			/* IEU1				*/
22:	andcc		%o2, 2, %g0			/* IEU1		Group		*/
	be,pn		%icc, 23f			/* CTI				*/

	 srlx		%g5, 16, %g4			/* IEU0				*/
	sth		%g4, [%o0]			/* Store	Group		*/
	add		%o0, 2, %o0			/* IEU0				*/
	mov		%g5, %g4			/* IEU1				*/

23:	srlx		%g4, 8, %g4			/* IEU0		Group		*/
	andcc		%o2, 1, %g0			/* IEU1				*/
	bne,a,pn	%icc, 24f			/* CTI				*/
	 stb		%g4, [%o0]			/* Store	Group		*/

	retl						/* CTI+IEU1	Group		*/
	 nop						/* IEU0				*/
24:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, 1, %o0			/* IEU0				*/

	.align		16
55:	sub		%o0, 1, %g6			/* IEU0		Group		*/
25:	andcc		%o0, 7, %g0			/* IEU1				*/
	be,a,pn		%icc, 4b			/* CTI				*/
	 andncc		%o2, 31, %g3			/* IEU1		Group		*/

	stb		%g0, [%o0]			/* Store	Group		*/
	subcc		%o2, 1, %o2			/* IEU1				*/
	bne,pt		%XCC, 25b			/* CTI				*/
	 add		%o0, 1, %o0			/* IEU0		Group		*/

	retl						/* CTI+IEU1	Group		*/
	 mov		%g6, %o0			/* IEU0				*/

	.align		16
26:	ldub		[%o1], %o3			/* Load				*/
	sllx		%g1, 7, %g2			/* IEU0		Group		*/
	stb		%o3, [%o0]			/* Store			*/
27:	subcc		%o2, 1, %o2			/* IEU1				*/

	be,pn		%XCC, 59b			/* CTI				*/
	 add		%o1, 1, %o1			/* IEU0		Group		*/
	add		%o0, 1, %o0			/* IEU1				*/
	andcc		%o3, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 55b			/* CTI				*/
	 lduba		[%o1] ASI_PNF, %o3		/* Load				*/
	andcc		%o0, 7, %g0			/* IEU1		Group		*/
	bne,a,pt	%icc, 27b			/* CTI				*/

	 stb		%o3, [%o0]			/* Store			*/
	andcc		%o1, 7, %g3			/* IEU1		Group		*/
	be,a,pt		%icc, 1b			/* CTI				*/
	 ldx		[%o1], %o3			/* Load				*/

28:	orcc		%g0, 64, %g4			/* IEU1		Group		*/
	sllx		%g3, 3, %g5			/* IEU0				*/
	sub		%g4, %g5, %g4			/* IEU0		Group		*/
	sub		%o1, %g3, %o1			/* IEU1				*/
							/* %g1 = 0101010101010101
							   %g2 = 8080808080808080
							   %g3 = source alignment
							   %g5 = number of bits to shift left
							   %g4 = number of bits to shift right */

	ldxa		[%o1] ASI_PNF, %o5		/* Load		Group		*/
	addcc		%o1, 8, %o1			/* IEU1				*/
29:	sllx		%o5, %g5, %o3			/* IEU0		Group		*/
	ldxa		[%o1] ASI_PNF, %o5		/* Load				*/

	subcc		%o2, 8, %o2			/* IEU1				*/
	bl,pn		%XCC, 17b			/* CTI				*/
	 srlx		%o5, %g4, %o4			/* IEU0		Group		*/
	add		%o1, 8, %o1			/* IEU1				*/

	or		%o3, %o4, %o3			/* IEU0		Group		*/
	add		%o0, 8, %o0			/* IEU1				*/
	sub		%o3, %g1, %o4			/* IEU0		Group		*/
#ifdef EIGHTBIT_NOT_RARE
	andn		%o4, %o3, %o4			/* IEU0		Group		*/
#endif
	andcc		%o4, %g2, %g0			/* IEU1		Group		*/

	be,a,pt		%xcc, 29b			/* CTI				*/
	 stx		%o3, [%o0-8]			/* Store			*/
	srlx		%o3, 56, %o4			/* IEU0		Group		*/
	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 36f			/* CTI				*/
	 srlx		%o3, 48, %g6			/* IEU0				*/
	andcc		%g6, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 35f			/* CTI				*/

	 srlx		%o3, 40, %o4			/* IEU0				*/
	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 34f			/* CTI				*/
	 srlx		%o3, 32, %g6			/* IEU0				*/

	andcc		%g6, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 33f			/* CTI				*/
	 srlx		%o3, 24, %o4			/* IEU0				*/
	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 32f			/* CTI				*/
	 srlx		%o3, 16, %g6			/* IEU0				*/
	andcc		%g6, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 31f			/* CTI				*/

	 srlx		%o3, 8, %o4			/* IEU0				*/
	andcc		%o4, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 30f			/* CTI				*/
	 andcc		%o3, 0xff, %g0			/* IEU1		Group		*/

	bne,pn		%icc, 29b			/* CTI				*/
	 stx		%o3, [%o0-8]			/* Store			*/
	sub		%o0, 1, %g6			/* IEU0		Group		*/
	ba,pt		%xcc, 4b			/* CTI				*/

	 andncc		%o2, 31, %g3			/* IEU1				*/
30:	subcc		%o0, 2, %g6			/* IEU0				*/
	ba,pt		%xcc, 3b			/* CTI				*/
	 sllx		%o4, 8, %g3			/* IEU0		Group		*/

31:	sllx		%g6, 16, %g3			/* IEU0		Group		*/
	ba,pt		%xcc, 3b			/* CTI				*/
	 sub		%o0, 3, %g6			/* IEU1				*/
32:	subcc		%o0, 4, %g6			/* IEU1		Group		*/

	ba,pt		%xcc, 3b			/* CTI				*/
	 sllx		%o4, 24, %g3			/* IEU0				*/
33:	sllx		%g6, 32, %g3			/* IEU0		Group		*/
	ba,pt		%xcc, 3b			/* CTI				*/

	 sub		%o0, 5, %g6			/* IEU1				*/
34:	subcc		%o0, 6, %g6			/* IEU1		Group		*/
	ba,pt		%xcc, 3b			/* CTI				*/
	 sllx		%o4, 40, %g3			/* IEU0				*/

35:	sllx		%g6, 48, %g3			/* IEU0		Group		*/
	ba,pt		%xcc, 3b			/* CTI				*/
	 sub		%o0, 7, %g6			/* IEU1				*/
36:	subcc		%o0, 8, %g6			/* IEU1		Group		*/

	ba,pt		%xcc, 3b			/* CTI				*/
	 sllx		%o4, 56, %g3			/* IEU0				*/
END(__stpncpy)

libc_hidden_def (__stpncpy)
weak_alias (__stpncpy, stpncpy)

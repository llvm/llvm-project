/* memchr (str, ch, n) -- Return pointer to first occurrence of CH in STR less
   than N.
   For SPARC v9.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jan Vondrák <jvon4518@ss1000.ms.mff.cuni.cz> and
                  Jakub Jelinek <jj@ultra.linux.cz>.
   This version is developed using the same algorithm as the fast C
   version which carries the following introduction:
   Based on strlen implementation by Torbjorn Granlund (tege@sics.se),
   with help from Dan Sahlin (dan@sics.se) and
   commentary by Jim Blandy (jimb@ai.mit.edu);
   adaptation to memchr suggested by Dick Karpinski (dick@cca.ucsf.edu),
   and implemented by Roland McGrath (roland@ai.mit.edu).

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
ENTRY(__memchr)
	and		%o1, 0xff, %o1			/* IEU0		Group		*/
#ifdef USE_BPR
	brz,pn		%o2, 12f			/* CTI+IEU1			*/
#else
	tst		%o2				/* IEU1				*/
	be,pn		%XCC, 12f			/* CTI				*/
#endif
	 sll		%o1, 8, %g3			/* IEU0		Group		*/
	addcc		%o0, %o2, %o2			/* IEU1				*/
	movcs		%XCC, -1, %o2			/* IEU0		Group		*/

	sethi		%hi(0x01010101), %g1		/* IEU0		Group		*/
	or		%g3, %o1, %g3			/* IEU1				*/
	ldub		[%o0], %o3			/* Load				*/
	sllx		%g3, 16, %g5			/* IEU0		Group		*/

	or		%g1, %lo(0x01010101), %g1	/* IEU1				*/
	sllx		%g1, 32, %g2			/* IEU0		Group		*/
	or		%g3, %g5, %g3			/* IEU1				*/
	sllx		%g3, 32, %g5			/* IEU0		Group		*/

	cmp		%o3, %o1			/* IEU1				*/
	be,pn		%xcc, 13f			/* CTI				*/
	 or		%g1, %g2, %g1			/* IEU0		Group		*/
	andcc		%o0, 7, %g0			/* IEU1				*/

	bne,a,pn	%icc, 21f			/* CTI				*/
	 add		%o0, 1, %o0			/* IEU0		Group		*/
	ldx		[%o0], %o3			/* Load		Group		*/
	sllx		%g1, 7, %g2			/* IEU0				*/

	or		%g3, %g5, %g3			/* IEU1				*/
1:	add		%o0, 8, %o0			/* IEU0		Group		*/
	xor		%o3, %g3, %o4			/* IEU1				*/
							/* %g1 = 0101010101010101	*
							 * %g2 = 8080088080808080	*
							 * %g3 =  c c c c c c c c	*
							 * %o3 =      value		*
							 * %o4 =   value XOR c		*/
2:	cmp		%o0, %o2			/* IEU1		Group		*/

	bgu,pn		%XCC, 11f			/* CTI				*/
	 ldxa		[%o0] ASI_PNF, %o3		/* Load				*/
	sub		%o4, %g1, %o5			/* IEU0		Group		*/
	add		%o0, 8, %o0			/* IEU1				*/
#ifdef EIGHTBIT_NOT_RARE
	andn		%o5, %o4, %o5			/* IEU0		Group		*/
#endif

	andcc		%o5, %g2, %g0			/* IEU1		Group		*/
	be,a,pt		%xcc, 2b			/* CTI				*/
	 xor		%o3, %g3, %o4			/* IEU0				*/
	srlx		%o4, 56, %g5			/* IEU0				*/

	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 3f			/* CTI				*/
	 srlx		%o4, 48, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 4f			/* CTI				*/
	 srlx		%o4, 40, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 5f			/* CTI				*/

	 srlx		%o4, 32, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 6f			/* CTI				*/
	 srlx		%o4, 24, %g5			/* IEU0				*/

	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 7f			/* CTI				*/
	 srlx		%o4, 16, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 8f			/* CTI				*/
	 srlx		%o4, 8, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 9f			/* CTI				*/

	 andcc		%o4, 0xff, %g0			/* IEU1		Group		*/
	bne,pt		%icc, 2b			/* CTI				*/
	 xor		%o3, %g3, %o4			/* IEU0				*/
	retl						/* CTI+IEU1	Group		*/

	 add		%o0, -9, %o0			/* IEU0				*/

	.align		16
3:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -16, %o0			/* IEU0				*/
4:   	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -15, %o0			/* IEU0				*/

5:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -14, %o0			/* IEU0				*/
6:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -13, %o0			/* IEU0				*/

7:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -12, %o0			/* IEU0				*/
8:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -11, %o0			/* IEU0				*/

9:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -10, %o0			/* IEU0				*/
11:	sub		%o4, %g1, %o5			/* IEU0		Group		*/
	sub		%o0, 8, %o0			/* IEU1				*/

	andcc		%o5, %g2, %g0			/* IEU1		Group		*/
	be,pt		%xcc, 12f			/* CTI				*/
	 sub		%o2, %o0, %o2			/* IEU0				*/
	tst		%o2				/* IEU1		Group		*/

	be,pn		%XCC, 12f			/* CTI				*/
	 srlx		%o4, 56, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 13f			/* CTI				*/

	 cmp		%o2, 1				/* IEU0				*/
	be,pn		%XCC, 12f			/* CTI		Group		*/
	 srlx		%o4, 48, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 14f			/* CTI				*/
	 cmp		%o2, 2				/* IEU1		Group		*/
	be,pn		%XCC, 12f			/* CTI				*/
	 srlx		%o4, 40, %g5			/* IEU0				*/

	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 15f			/* CTI				*/
	 cmp		%o2, 3				/* IEU1		Group		*/
	be,pn		%XCC, 12f			/* CTI				*/

	 srlx		%o4, 32, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 16f			/* CTI				*/
	 cmp		%o2, 4				/* IEU1		Group		*/

	be,pn		%XCC, 12f			/* CTI				*/
	 srlx		%o4, 24, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 17f			/* CTI				*/

	 cmp		%o2, 5				/* IEU1		Group		*/
	be,pn		%XCC, 12f			/* CTI				*/
	 srlx		%o4, 16, %g5			/* IEU0				*/
	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 18f			/* CTI				*/
	 cmp		%o2, 6				/* IEU1		Group		*/
	be,pn		%XCC, 12f			/* CTI				*/
	 srlx		%o4, 8, %g5			/* IEU0				*/

	andcc		%g5, 0xff, %g0			/* IEU1		Group		*/
	be,pn		%icc, 19f			/* CTI				*/
	 nop						/* IEU0				*/
12:	retl						/* CTI+IEU1	Group		*/

	 clr		%o0				/* IEU0				*/
	nop						/* Stub				*/
13:	retl						/* CTI+IEU1	Group		*/
	 nop						/* IEU0				*/

14:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, 1, %o0			/* IEU0				*/
15:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, 2, %o0			/* IEU0				*/

16:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, 3, %o0			/* IEU0				*/
17:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, 4, %o0			/* IEU0				*/

18:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, 5, %o0			/* IEU0				*/
19:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, 6, %o0			/* IEU0				*/

21:	cmp		%o0, %o2			/* IEU1				*/
	be,pn		%XCC, 12b			/* CTI				*/
	 sllx		%g1, 7, %g2			/* IEU0		Group		*/
	ldub		[%o0], %o3			/* Load				*/

	or		%g3, %g5, %g3			/* IEU1				*/
22:	andcc		%o0, 7, %g0			/* IEU1		Group		*/
	be,a,pn		%icc, 1b			/* CTI				*/
	 ldx		[%o0], %o3			/* Load				*/

	cmp		%o3, %o1			/* IEU1		Group		*/
	be,pn		%xcc, 23f			/* CTI				*/
	 add		%o0, 1, %o0			/* IEU0				*/
	cmp		%o0, %o2			/* IEU1		Group		*/

	bne,a,pt	%XCC, 22b			/* CTI				*/
	 ldub		[%o0], %o3			/* Load				*/
	retl						/* CTI+IEU1	Group		*/
	 clr		%o0				/* IEU0				*/

23:	retl						/* CTI+IEU1	Group		*/
	 add		%o0, -1, %o0			/* IEU0				*/
END(__memchr)

weak_alias (__memchr, memchr)
libc_hidden_builtin_def (memchr)

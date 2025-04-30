/* rawmemchr (str, ch) -- Return pointer to first occurrence of CH in STR.
   For SPARC v9.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jj@ultra.linux.cz>.
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
ENTRY(__rawmemchr)
	and		%o1, 0xff, %o1			/* IEU0		Group		*/
	sethi		%hi(0x01010101), %g1		/* IEU1				*/
	ldub		[%o0], %o3			/* Load				*/
	sll		%o1, 8, %o4			/* IEU0		Group		*/

	or		%g1, %lo(0x01010101), %g1	/* IEU1				*/
	sllx		%g1, 32, %g2			/* IEU0		Group		*/
	or		%o4, %o1, %o4			/* IEU1				*/
	andcc		%o0, 7, %g0			/* IEU1		Group		*/

	sll		%o4, 16, %g5			/* IEU0				*/
	or		%o4, %g5, %o4			/* IEU0		Group		*/
	or		%g1, %g2, %g1			/* IEU1				*/
	bne,pn		%icc, 32f			/* CTI				*/

	 sllx		%o4, 32, %g5			/* IEU0		Group		*/
	cmp		%o3, %o1			/* IEU1				*/
	be,pn		%icc, 30f			/* CTI				*/
	 sllx		%g1, 7, %g2			/* IEU0		Group		*/

18:	ldx		[%o0], %o3			/* Load				*/
	or		%o4, %g5, %o4			/* IEU1				*/
	add		%o0, 8, %o0			/* IEU0		Group		*/
19:	xor		%o3, %o4, %o3			/* IEU0		Group		*/

	sub		%o3, %g1, %o2			/* IEU0		Group		*/
#ifdef EIGHTBIT_NOT_RARE
	andn		%o2, %o3, %o5			/* IEU0		Group		*/
	ldxa		[%o0] ASI_PNF, %o3		/* Load				*/
	andcc		%o5, %g2, %g0			/* IEU1		Group		*/
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
	 xor		%o3, %o4, %o3			/* IEU0				*/
	andcc		%g3, 0xff, %g0			/* IEU1		Group		*/

	be,pn		%icc, 22f			/* CTI				*/
	 sub		%o3, %g1, %o2			/* IEU0				*/
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
	 sllx		%g1, 7, %g2			/* IEU0				*/
	add		%o0, 1, %o0			/* IEU0		Group		*/

	cmp		%o3, %o1			/* IEU1				*/
	bne,a,pt	%icc, 32b			/* CTI				*/
	 lduba		[%o0] ASI_PNF, %o3		/* Load				*/
	retl						/* CTI+IEU1	Group		*/

	 add		%o0, -1, %o0			/* IEU0				*/
END(__rawmemchr)

libc_hidden_def (__rawmemchr)
weak_alias (__rawmemchr, rawmemchr)

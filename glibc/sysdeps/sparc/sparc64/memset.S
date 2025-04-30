/* Set a block of memory to some byte value.
   For UltraSPARC.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David S. Miller (davem@caip.rutgers.edu) and
                  Jakub Jelinek (jj@ultra.linux.cz).

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
#endif
#define FPRS_FEF	4

#define SET_BLOCKS(base, offset, source)		\
	stx		source, [base - offset - 0x18];	\
	stx		source, [base - offset - 0x10];	\
	stx		source, [base - offset - 0x08];	\
	stx		source, [base - offset - 0x00];

	/* Well, memset is a lot easier to get right than bcopy... */
	.text
	.align		32
ENTRY(memset)
	andcc		%o1, 0xff, %o1
	mov		%o0, %o5
	be,a,pt		%icc, 50f
#ifndef USE_BPR
	 srl		%o2, 0, %o1
#else
	 mov		%o2, %o1
#endif
	cmp		%o2, 7
#ifndef USE_BPR
	srl		%o2, 0, %o2
#endif
	bleu,pn		%XCC, 17f
	 andcc		%o0, 3, %g5
	be,pt		%xcc, 4f
	 and		%o1, 0xff, %o1
	cmp		%g5, 3
	be,pn		%xcc, 2f
	 stb		%o1, [%o0 + 0x00]
	cmp		%g5, 2
	be,pt		%xcc, 2f
	 stb		%o1, [%o0 + 0x01]
	stb		%o1, [%o0 + 0x02]
2:	sub		%g5, 4, %g5
	sub		%o0, %g5, %o0
	add		%o2, %g5, %o2
4:	sllx		%o1, 8, %g1
	andcc		%o0, 4, %g0
	or		%o1, %g1, %o1
	sllx		%o1, 16, %g1
	or		%o1, %g1, %o1
	be,pt		%xcc, 2f
	 sllx		%o1, 32, %g1
	stw		%o1, [%o0]
	sub		%o2, 4, %o2
	add		%o0, 4, %o0
2:	cmp		%o2, 128
	or		%o1, %g1, %o1
	blu,pn		%xcc, 9f
	 andcc		%o0, 0x38, %g5
	be,pn		%icc, 6f
	 mov		64, %o4
	andcc		%o0, 8, %g0
	be,pn		%icc, 1f
	 sub		%o4, %g5, %o4
	stx		%o1, [%o0]
	add		%o0, 8, %o0
1:	andcc		%o4, 16, %g0
	be,pn		%icc, 1f
	 sub		%o2, %o4, %o2
	stx		%o1, [%o0]
	stx		%o1, [%o0 + 8]
	add		%o0, 16, %o0
1:	andcc		%o4, 32, %g0
	be,pn		%icc, 7f
	 andncc		%o2, 0x3f, %o3
	stw		%o1, [%o0]
	stw		%o1, [%o0 + 4]
	stw		%o1, [%o0 + 8]
	stw		%o1, [%o0 + 12]
	stw		%o1, [%o0 + 16]
	stw		%o1, [%o0 + 20]
	stw		%o1, [%o0 + 24]
	stw		%o1, [%o0 + 28]
	add		%o0, 32, %o0
7:	be,pn		%xcc, 9f
	 nop
	ldd		[%o0 - 8], %f0
18:	wr		%g0, ASI_BLK_P, %asi
	membar		#StoreStore | #LoadStore
	andcc		%o3, 0xc0, %g5
	and		%o2, 0x3f, %o2
	fsrc2		%f0, %f2
	fsrc2		%f0, %f4
	andn		%o3, 0xff, %o3
	fsrc2		%f0, %f6
	cmp		%g5, 64
	fsrc2		%f0, %f8
	fsrc2		%f0, %f10
	fsrc2		%f0, %f12
	brz,pn		%g5, 10f
	 fsrc2		%f0, %f14
	be,pn		%icc, 2f
	 stda		%f0, [%o0 + 0x00] %asi
	cmp		%g5, 128
	be,pn		%icc, 2f
	 stda		%f0, [%o0 + 0x40] %asi
	stda		%f0, [%o0 + 0x80] %asi
2:	brz,pn		%o3, 12f
	 add		%o0, %g5, %o0
10:	stda		%f0, [%o0 + 0x00] %asi
	stda		%f0, [%o0 + 0x40] %asi
	stda		%f0, [%o0 + 0x80] %asi
	stda		%f0, [%o0 + 0xc0] %asi
11:	subcc		%o3, 256, %o3
	bne,pt		%xcc, 10b
	 add		%o0, 256, %o0
12:	wr		%g0, FPRS_FEF, %fprs
	membar		#StoreLoad | #StoreStore
9:	andcc		%o2, 0x78, %g5
	be,pn		%xcc, 13f
	 andcc		%o2, 7, %o2
14:	rd		%pc, %o4
	srl		%g5, 1, %o3
	sub		%o4, %o3, %o4
	jmpl		%o4 + (13f - 14b), %g0
	 add		%o0, %g5, %o0
12:	SET_BLOCKS	(%o0, 0x68, %o1)
	SET_BLOCKS	(%o0, 0x48, %o1)
	SET_BLOCKS	(%o0, 0x28, %o1)
	SET_BLOCKS	(%o0, 0x08, %o1)
13:	be,pn		%xcc, 8f
	 andcc		%o2, 4, %g0
	be,pn		%xcc, 1f
	 andcc		%o2, 2, %g0
	stw		%o1, [%o0]
	add		%o0, 4, %o0
1:	be,pn		%xcc, 1f
	 andcc		%o2, 1, %g0
	sth		%o1, [%o0]
	add		%o0, 2, %o0
1:	bne,a,pn	%xcc, 8f
	 stb		%o1, [%o0]
8:	retl
	 mov		%o5, %o0
17:	brz,pn		%o2, 0f
8:	 add		%o0, 1, %o0
	subcc		%o2, 1, %o2
	bne,pt		%xcc, 8b
	 stb		%o1, [%o0 - 1]
0:	retl
	 mov		%o5, %o0

6:	stx		%o1, [%o0]
	andncc		%o2, 0x3f, %o3
	be,pn		%xcc, 9b
	 nop
	ba,pt		%xcc, 18b
	 ldd		[%o0], %f0
END(memset)
libc_hidden_builtin_def (memset)

#define ZERO_BLOCKS(base, offset, source)		\
	stx		source, [base - offset - 0x38];	\
	stx		source, [base - offset - 0x30];	\
	stx		source, [base - offset - 0x28];	\
	stx		source, [base - offset - 0x20];	\
	stx		source, [base - offset - 0x18];	\
	stx		source, [base - offset - 0x10];	\
	stx		source, [base - offset - 0x08];	\
	stx		source, [base - offset - 0x00];

	.text
	.align		32
ENTRY(__bzero)
#ifndef USE_BPR
	srl		%o1, 0, %o1
#endif
	mov		%o0, %o5
50:	cmp		%o1, 7
	bleu,pn		%xcc, 17f
	 andcc		%o0, 3, %o2
	be,a,pt		%xcc, 4f
	 andcc		%o0, 4, %g0
	cmp		%o2, 3
	be,pn		%xcc, 2f
	 stb		%g0, [%o0 + 0x00]
	cmp		%o2, 2
	be,pt		%xcc, 2f
	 stb		%g0, [%o0 + 0x01]
	stb		%g0, [%o0 + 0x02]
2:	sub		%o2, 4, %o2
	sub		%o0, %o2, %o0
	add		%o1, %o2, %o1
	andcc		%o0, 4, %g0
4:	be,pt		%xcc, 2f
	 cmp		%o1, 128
	stw		%g0, [%o0]
	sub		%o1, 4, %o1
	add		%o0, 4, %o0
2:	blu,pn		%xcc, 9f
	 andcc		%o0, 0x38, %o2
	be,pn		%icc, 6f
	 mov		64, %o4
	andcc		%o0, 8, %g0
	be,pn		%icc, 1f
	 sub		%o4, %o2, %o4
	stx		%g0, [%o0]
	add		%o0, 8, %o0
1:	andcc		%o4, 16, %g0
	be,pn		%icc, 1f
	 sub		%o1, %o4, %o1
	stx		%g0, [%o0]
	stx		%g0, [%o0 + 8]
	add		%o0, 16, %o0
1:	andcc		%o4, 32, %g0
	be,pn		%icc, 7f
	 andncc		%o1, 0x3f, %o3
	stx		%g0, [%o0]
	stx		%g0, [%o0 + 8]
	stx		%g0, [%o0 + 16]
	stx		%g0, [%o0 + 24]
	add		%o0, 32, %o0
6:	andncc		%o1, 0x3f, %o3
7:	be,pn		%xcc, 9f
	 wr		%g0, ASI_BLK_P, %asi
	membar		#StoreLoad | #StoreStore | #LoadStore
	fzero		%f0
	andcc		%o3, 0xc0, %o2
	and		%o1, 0x3f, %o1
	fzero		%f2
	andn		%o3, 0xff, %o3
	faddd		%f0, %f2, %f4
	fmuld		%f0, %f2, %f6
	cmp		%o2, 64
	faddd		%f0, %f2, %f8
	fmuld		%f0, %f2, %f10
	faddd		%f0, %f2, %f12
	brz,pn		%o2, 10f
	 fmuld		%f0, %f2, %f14
	be,pn		%icc, 2f
	 stda		%f0, [%o0 + 0x00] %asi
	cmp		%o2, 128
	be,pn		%icc, 2f
	 stda		%f0, [%o0 + 0x40] %asi
	stda		%f0, [%o0 + 0x80] %asi
2:	brz,pn		%o3, 12f
	 add		%o0, %o2, %o0
10:	stda		%f0, [%o0 + 0x00] %asi
	stda		%f0, [%o0 + 0x40] %asi
	stda		%f0, [%o0 + 0x80] %asi
	stda		%f0, [%o0 + 0xc0] %asi
11:	subcc		%o3, 256, %o3
	bne,pt		%xcc, 10b
	 add		%o0, 256, %o0
12:	wr		%g0, FPRS_FEF, %fprs
	membar		#StoreLoad | #StoreStore
9:	andcc		%o1, 0xf8, %o2
	be,pn		%xcc, 13f
	 andcc		%o1, 7, %o1
14:	rd		%pc, %o4
	srl		%o2, 1, %o3
	sub		%o4, %o3, %o4
	jmpl		%o4 + (13f - 14b), %g0
	 add		%o0, %o2, %o0
12:	ZERO_BLOCKS	(%o0, 0xc8, %g0)
	ZERO_BLOCKS	(%o0, 0x88, %g0)
	ZERO_BLOCKS	(%o0, 0x48, %g0)
	ZERO_BLOCKS	(%o0, 0x08, %g0)
13:	be,pn		%xcc, 8f
	 andcc		%o1, 4, %g0
	be,pn		%xcc, 1f
	 andcc		%o1, 2, %g0
	stw		%g0, [%o0]
	add		%o0, 4, %o0
1:	be,pn		%xcc, 1f
	 andcc		%o1, 1, %g0
	sth		%g0, [%o0]
	add		%o0, 2, %o0
1:	bne,a,pn	%xcc, 8f
	 stb		%g0, [%o0]
8:	retl
	 mov		%o5, %o0
17:	be,pn		%xcc, 13b
	 orcc		%o1, 0, %g0
	be,pn		%xcc, 0f
8:	 add		%o0, 1, %o0
	subcc		%o1, 1, %o1
	bne,pt		%xcc, 8b
	 stb		%g0, [%o0 - 1]
0:	retl
	 mov		%o5, %o0
END(__bzero)

weak_alias (__bzero, bzero)

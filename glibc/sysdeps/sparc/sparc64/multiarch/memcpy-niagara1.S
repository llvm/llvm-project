/* Copy SIZE bytes from SRC to DEST.  For SUN4V Niagara.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David S. Miller (davem@davemloft.net)

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

#define ASI_BLK_INIT_QUAD_LDD_P	0xe2
#define ASI_P			0x80
#define ASI_PNF			0x82

#define LOAD(type,addr,dest)	type##a [addr] ASI_P, dest
#define LOAD_TWIN(addr_reg,dest0,dest1)	\
	ldda [addr_reg] ASI_BLK_INIT_QUAD_LDD_P, dest0

#define STORE(type,src,addr)	type src, [addr]
#define STORE_INIT(src,addr)	stxa src, [addr] %asi

#ifndef XCC
#define USE_BPR
#define XCC xcc
#endif

#if IS_IN (libc)

	.register	%g2,#scratch
	.register	%g3,#scratch
	.register	%g6,#scratch

	.text

ENTRY(__mempcpy_niagara1)
	ba,pt		%XCC, 101f
	 add		%o0, %o2, %g5
END(__mempcpy_niagara1)

	.align		32
ENTRY(__memcpy_niagara1)
100:	/* %o0=dst, %o1=src, %o2=len */
	mov		%o0, %g5
101:
# ifndef USE_BPR
	srl		%o2, 0, %o2
# endif
	cmp		%o2, 0
	be,pn		%XCC, 85f
218:	 or		%o0, %o1, %o3
	cmp		%o2, 16
	blu,a,pn	%XCC, 80f
	 or		%o3, %o2, %o3

	/* 2 blocks (128 bytes) is the minimum we can do the block
	 * copy with.  We need to ensure that we'll iterate at least
	 * once in the block copy loop.  At worst we'll need to align
	 * the destination to a 64-byte boundary which can chew up
	 * to (64 - 1) bytes from the length before we perform the
	 * block copy loop.
	 */
	cmp		%o2, (2 * 64)
	blu,pt		%XCC, 70f
	 andcc		%o3, 0x7, %g0

	/* %o0:	dst
	 * %o1:	src
	 * %o2:	len  (known to be >= 128)
	 *
	 * The block copy loops will use %o4/%o5,%g2/%g3 as
	 * temporaries while copying the data.
	 */

	LOAD(prefetch, %o1, #one_read)
	wr		%g0, ASI_BLK_INIT_QUAD_LDD_P, %asi

	/* Align destination on 64-byte boundary.  */
	andcc		%o0, (64 - 1), %o4
	be,pt		%XCC, 2f
	 sub		%o4, 64, %o4
	sub		%g0, %o4, %o4	! bytes to align dst
	sub		%o2, %o4, %o2
1:	subcc		%o4, 1, %o4
	LOAD(ldub, %o1, %g1)
	STORE(stb, %g1, %o0)
	add		%o1, 1, %o1
	bne,pt		%XCC, 1b
	add		%o0, 1, %o0

	/* If the source is on a 16-byte boundary we can do
	 * the direct block copy loop.  If it is 8-byte aligned
	 * we can do the 16-byte loads offset by -8 bytes and the
	 * init stores offset by one register.
	 *
	 * If the source is not even 8-byte aligned, we need to do
	 * shifting and masking (basically integer faligndata).
	 *
	 * The careful bit with init stores is that if we store
	 * to any part of the cache line we have to store the whole
	 * cacheline else we can end up with corrupt L2 cache line
	 * contents.  Since the loop works on 64-bytes of 64-byte
	 * aligned store data at a time, this is easy to ensure.
	 */
2:
	andcc		%o1, (16 - 1), %o4
	andn		%o2, (64 - 1), %g1	! block copy loop iterator
	sub		%o2, %g1, %o2		! final sub-block copy bytes
	be,pt		%XCC, 50f
	 cmp		%o4, 8
	be,a,pt		%XCC, 10f
	 sub		%o1, 0x8, %o1

	/* Neither 8-byte nor 16-byte aligned, shift and mask.  */
	mov		%g1, %o4
	and		%o1, 0x7, %g1
	sll		%g1, 3, %g1
	mov		64, %o3
	andn		%o1, 0x7, %o1
	LOAD(ldx, %o1, %g2)
	sub		%o3, %g1, %o3
	sllx		%g2, %g1, %g2

#define SWIVEL_ONE_DWORD(SRC, TMP1, TMP2, PRE_VAL, PRE_SHIFT, POST_SHIFT, DST)\
	LOAD(ldx, SRC, TMP1); \
	srlx		TMP1, PRE_SHIFT, TMP2; \
	or		TMP2, PRE_VAL, TMP2; \
	STORE_INIT(TMP2, DST); \
	sllx		TMP1, POST_SHIFT, PRE_VAL;

1:	add		%o1, 0x8, %o1
	SWIVEL_ONE_DWORD(%o1, %g3, %o5, %g2, %o3, %g1, %o0 + 0x00)
	add		%o1, 0x8, %o1
	SWIVEL_ONE_DWORD(%o1, %g3, %o5, %g2, %o3, %g1, %o0 + 0x08)
	add		%o1, 0x8, %o1
	SWIVEL_ONE_DWORD(%o1, %g3, %o5, %g2, %o3, %g1, %o0 + 0x10)
	add		%o1, 0x8, %o1
	SWIVEL_ONE_DWORD(%o1, %g3, %o5, %g2, %o3, %g1, %o0 + 0x18)
	add		%o1, 32, %o1
	LOAD(prefetch, %o1, #one_read)
	sub		%o1, 32 - 8, %o1
	SWIVEL_ONE_DWORD(%o1, %g3, %o5, %g2, %o3, %g1, %o0 + 0x20)
	add		%o1, 8, %o1
	SWIVEL_ONE_DWORD(%o1, %g3, %o5, %g2, %o3, %g1, %o0 + 0x28)
	add		%o1, 8, %o1
	SWIVEL_ONE_DWORD(%o1, %g3, %o5, %g2, %o3, %g1, %o0 + 0x30)
	add		%o1, 8, %o1
	SWIVEL_ONE_DWORD(%o1, %g3, %o5, %g2, %o3, %g1, %o0 + 0x38)
	subcc		%o4, 64, %o4
	bne,pt		%XCC, 1b
	 add		%o0, 64, %o0

#undef SWIVEL_ONE_DWORD

	srl		%g1, 3, %g1
	ba,pt		%XCC, 60f
	 add		%o1, %g1, %o1

10:	/* Destination is 64-byte aligned, source was only 8-byte
	 * aligned but it has been subtracted by 8 and we perform
	 * one twin load ahead, then add 8 back into source when
	 * we finish the loop.
	 */
	LOAD_TWIN(%o1, %o4, %o5)
1:	add		%o1, 16, %o1
	LOAD_TWIN(%o1, %g2, %g3)
	add		%o1, 16 + 32, %o1
	LOAD(prefetch, %o1, #one_read)
	sub		%o1, 32, %o1
	STORE_INIT(%o5, %o0 + 0x00)		! initializes cache line
	STORE_INIT(%g2, %o0 + 0x08)
	LOAD_TWIN(%o1, %o4, %o5)
	add		%o1, 16, %o1
	STORE_INIT(%g3, %o0 + 0x10)
	STORE_INIT(%o4, %o0 + 0x18)
	LOAD_TWIN(%o1, %g2, %g3)
	add		%o1, 16, %o1
	STORE_INIT(%o5, %o0 + 0x20)
	STORE_INIT(%g2, %o0 + 0x28)
	LOAD_TWIN(%o1, %o4, %o5)
	STORE_INIT(%g3, %o0 + 0x30)
	STORE_INIT(%o4, %o0 + 0x38)
	subcc		%g1, 64, %g1
	bne,pt		%XCC, 1b
	 add		%o0, 64, %o0

	ba,pt		%XCC, 60f
	 add		%o1, 0x8, %o1

50:	/* Destination is 64-byte aligned, and source is 16-byte
	 * aligned.
	 */
1:	LOAD_TWIN(%o1, %o4, %o5)
	add	%o1, 16, %o1
	LOAD_TWIN(%o1, %g2, %g3)
	add	%o1, 16 + 32, %o1
	LOAD(prefetch, %o1, #one_read)
	sub	%o1, 32, %o1
	STORE_INIT(%o4, %o0 + 0x00)		! initializes cache line
	STORE_INIT(%o5, %o0 + 0x08)
	LOAD_TWIN(%o1, %o4, %o5)
	add	%o1, 16, %o1
	STORE_INIT(%g2, %o0 + 0x10)
	STORE_INIT(%g3, %o0 + 0x18)
	LOAD_TWIN(%o1, %g2, %g3)
	add	%o1, 16, %o1
	STORE_INIT(%o4, %o0 + 0x20)
	STORE_INIT(%o5, %o0 + 0x28)
	STORE_INIT(%g2, %o0 + 0x30)
	STORE_INIT(%g3, %o0 + 0x38)
	subcc	%g1, 64, %g1
	bne,pt	%XCC, 1b
	 add	%o0, 64, %o0
	/* fall through */

60:
	/* %o2 contains any final bytes still needed to be copied
	 * over. If anything is left, we copy it one byte at a time.
	 */
	wr		%g0, ASI_PNF, %asi
	brz,pt		%o2, 85f
	 sub		%o0, %o1, %o3
	ba,a,pt		%XCC, 90f

	.align		64
70: /* 16 < len <= 64 */
	bne,pn		%XCC, 75f
	 sub		%o0, %o1, %o3

72:
	andn		%o2, 0xf, %o4
	and		%o2, 0xf, %o2
1:	subcc		%o4, 0x10, %o4
	LOAD(ldx, %o1, %o5)
	add		%o1, 0x08, %o1
	LOAD(ldx, %o1, %g1)
	sub		%o1, 0x08, %o1
	STORE(stx, %o5, %o1 + %o3)
	add		%o1, 0x8, %o1
	STORE(stx, %g1, %o1 + %o3)
	bgu,pt		%XCC, 1b
	 add		%o1, 0x8, %o1
73:	andcc		%o2, 0x8, %g0
	be,pt		%XCC, 1f
	 nop
	sub		%o2, 0x8, %o2
	LOAD(ldx, %o1, %o5)
	STORE(stx, %o5, %o1 + %o3)
	add		%o1, 0x8, %o1
1:	andcc		%o2, 0x4, %g0
	be,pt		%XCC, 1f
	 nop
	sub		%o2, 0x4, %o2
	LOAD(lduw, %o1, %o5)
	STORE(stw, %o5, %o1 + %o3)
	add		%o1, 0x4, %o1
1:	cmp		%o2, 0
	be,pt		%XCC, 85f
	 nop
	ba,pt		%XCC, 90f
	 nop

75:
	andcc		%o0, 0x7, %g1
	sub		%g1, 0x8, %g1
	be,pn		%icc, 2f
	 sub		%g0, %g1, %g1
	sub		%o2, %g1, %o2

1:	subcc		%g1, 1, %g1
	LOAD(ldub, %o1, %o5)
	STORE(stb, %o5, %o1 + %o3)
	bgu,pt		%icc, 1b
	 add		%o1, 1, %o1

2:	add		%o1, %o3, %o0
	andcc		%o1, 0x7, %g1
	bne,pt		%icc, 8f
	 sll		%g1, 3, %g1

	cmp		%o2, 16
	bgeu,pt		%icc, 72b
	 nop
	ba,a,pt		%XCC, 73b

8:	mov		64, %o3
	andn		%o1, 0x7, %o1
	LOAD(ldx, %o1, %g2)
	sub		%o3, %g1, %o3
	andn		%o2, 0x7, %o4
	sllx		%g2, %g1, %g2
1:	add		%o1, 0x8, %o1
	LOAD(ldx, %o1, %g3)
	subcc		%o4, 0x8, %o4
	srlx		%g3, %o3, %o5
	or		%o5, %g2, %o5
	STORE(stx, %o5, %o0)
	add		%o0, 0x8, %o0
	bgu,pt		%icc, 1b
	 sllx		%g3, %g1, %g2

	srl		%g1, 3, %g1
	andcc		%o2, 0x7, %o2
	be,pn		%icc, 85f
	 add		%o1, %g1, %o1
	ba,pt		%XCC, 90f
	 sub		%o0, %o1, %o3

	.align		64
80: /* 0 < len <= 16 */
	andcc		%o3, 0x3, %g0
	bne,pn		%XCC, 90f
	 sub		%o0, %o1, %o3

1:
	subcc		%o2, 4, %o2
	LOAD(lduw, %o1, %g1)
	STORE(stw, %g1, %o1 + %o3)
	bgu,pt		%XCC, 1b
	 add		%o1, 4, %o1

85:	retl
	 mov		%g5, %o0

	.align		32
90:
	subcc		%o2, 1, %o2
	LOAD(ldub, %o1, %g1)
	STORE(stb, %g1, %o1 + %o3)
	bgu,pt		%XCC, 90b
	 add		%o1, 1, %o1
	retl
	 mov		%g5, %o0

END(__memcpy_niagara1)

#endif

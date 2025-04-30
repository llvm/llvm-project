/* Optimized strlen implementation for POWER10 LE.
   Copyright (C) 2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

/* To reuse the code for rawmemchr, we have some extra steps compared to the
   strlen implementation:
      - Sum the initial value of r3 with the position at which the char was
        found, to guarantee we return a pointer and not the length.
      - In the main loop, subtract each byte by the char we are looking for,
        so we can keep using vminub to quickly check 64B at once.  */
#ifdef USE_AS_RAWMEMCHR
# ifndef RAWMEMCHR
#  define FUNCNAME __rawmemchr
# else
#  define FUNCNAME RAWMEMCHR
# endif
# define MCOUNT_NARGS 2
# define VREG_ZERO v20
# define OFF_START_LOOP 256
# define RAWMEMCHR_SUBTRACT_VECTORS \
	vsububm   v4,v4,v18;	    \
	vsububm   v5,v5,v18;	    \
	vsububm   v6,v6,v18;	    \
	vsububm   v7,v7,v18;
# define TAIL(vreg,increment)	   \
	vctzlsbb  r4,vreg;	   \
	addi	  r4,r4,increment; \
	add	  r3,r5,r4;	   \
	blr

#else /* strlen */

# ifndef STRLEN
#  define FUNCNAME __strlen
#  define DEFINE_STRLEN_HIDDEN_DEF 1
# else
#  define FUNCNAME STRLEN
# endif
# define MCOUNT_NARGS 1
# define VREG_ZERO v18
# define OFF_START_LOOP 192
# define TAIL(vreg,increment)	   \
	vctzlsbb  r4,vreg;	   \
	subf	  r3,r3,r5;	   \
	addi	  r4,r4,increment; \
	add	  r3,r3,r4;	   \
	blr
#endif /* USE_AS_RAWMEMCHR */

/* TODO: Replace macros by the actual instructions when minimum binutils becomes
   >= 2.35.  This is used to keep compatibility with older versions.  */
#define VEXTRACTBM(rt,vrb)	 \
	.long(((4)<<(32-6))	 \
	      | ((rt)<<(32-11))	 \
	      | ((8)<<(32-16))	 \
	      | ((vrb)<<(32-21)) \
	      | 1602)

#define LXVP(xtp,dq,ra)		   \
	.long(((6)<<(32-6))		   \
	      | ((((xtp)-32)>>1)<<(32-10)) \
	      | ((1)<<(32-11))		   \
	      | ((ra)<<(32-16))		   \
	      | dq)

#define CHECK16(vreg,offset,addr,label) \
	lxv	  vreg+32,offset(addr);	\
	vcmpequb. vreg,vreg,v18;	\
	bne	  cr6,L(label);

/* Load 4 quadwords, merge into one VR for speed and check for NULLs.  r6 has #
   of bytes already checked.  */
#define CHECK64(offset,addr,label)	    \
	li	  r6,offset;		    \
	LXVP(v4+32,offset,addr);	    \
	LXVP(v6+32,offset+32,addr);	    \
	RAWMEMCHR_SUBTRACT_VECTORS;	    \
	vminub	  v14,v4,v5;		    \
	vminub	  v15,v6,v7;		    \
	vminub	  v16,v14,v15;		    \
	vcmpequb. v0,v16,VREG_ZERO;	    \
	bne	  cr6,L(label)

/* Implements the function

   int [r3] strlen (const void *s [r3])

   but when USE_AS_RAWMEMCHR is set, implements the function

   void* [r3] rawmemchr (const void *s [r3], int c [r4])

   The implementation can load bytes past a matching byte, but only
   up to the next 64B boundary, so it never crosses a page.  */

.machine power9

ENTRY_TOCLESS (FUNCNAME, 4)
	CALL_MCOUNT MCOUNT_NARGS

#ifdef USE_AS_RAWMEMCHR
	xori	r5,r4,0xff

	mtvsrd	v18+32,r4	/* matching char in v18  */
	mtvsrd	v19+32,r5	/* non matching char in v19  */

	vspltb	v18,v18,7	/* replicate  */
	vspltb	v19,v19,7	/* replicate  */
#else
	vspltisb  v19,-1
#endif
	vspltisb  VREG_ZERO,0

	/* Next 16B-aligned address. Prepare address for L(aligned).  */
	addi	  r5,r3,16
	clrrdi	  r5,r5,4

	/* Align data and fill bytes not loaded with non matching char.	 */
	lvx	  v0,0,r3
	lvsr	  v1,0,r3
	vperm	  v0,v19,v0,v1

	vcmpequb. v6,v0,v18
	beq	  cr6,L(aligned)

#ifdef USE_AS_RAWMEMCHR
	vctzlsbb  r6,v6
	add	  r3,r3,r6
#else
	vctzlsbb  r3,v6
#endif
	blr

	/* Test up to OFF_START_LOOP-16 bytes in 16B chunks.  The main loop is
	   optimized for longer strings, so checking the first bytes in 16B
	   chunks benefits a lot small strings.  */
	.p2align 5
L(aligned):
#ifdef USE_AS_RAWMEMCHR
	cmpdi	cr5,r4,0	/* Check if c == 0.  This will be useful to
				  choose how we will perform the main loop.  */
#endif
	/* Prepare address for the loop.  */
	addi	  r4,r3,OFF_START_LOOP
	clrrdi	  r4,r4,6

	CHECK16(v0,0,r5,tail1)
	CHECK16(v1,16,r5,tail2)
	CHECK16(v2,32,r5,tail3)
	CHECK16(v3,48,r5,tail4)
	CHECK16(v4,64,r5,tail5)
	CHECK16(v5,80,r5,tail6)
	CHECK16(v6,96,r5,tail7)
	CHECK16(v7,112,r5,tail8)
	CHECK16(v8,128,r5,tail9)
	CHECK16(v9,144,r5,tail10)
	CHECK16(v10,160,r5,tail11)
#ifdef USE_AS_RAWMEMCHR
	CHECK16(v0,176,r5,tail12)
	CHECK16(v1,192,r5,tail13)
	CHECK16(v2,208,r5,tail14)
	CHECK16(v3,224,r5,tail15)
#endif

	addi	  r5,r4,128

#ifdef USE_AS_RAWMEMCHR
	/* If c == 0, use the same loop as strlen, without the vsububm.  */
	beq	cr5,L(loop)

	/* This is very similar to the block after L(loop), the difference is
	   that here RAWMEMCHR_SUBTRACT_VECTORS is not empty, and we subtract
	   each byte loaded by the char we are looking for, this way we can keep
	   using vminub to merge the results and checking for nulls.  */
	.p2align 5
L(rawmemchr_loop):
	CHECK64(0,r4,pre_tail_64b)
	CHECK64(64,r4,pre_tail_64b)
	addi	  r4,r4,256

	CHECK64(0,r5,tail_64b)
	CHECK64(64,r5,tail_64b)
	addi	  r5,r5,256

	b	  L(rawmemchr_loop)
#endif
	/* Switch to a more aggressive approach checking 64B each time.  Use 2
	   pointers 128B apart and unroll the loop once to make the pointer
	   updates and usages separated enough to avoid stalls waiting for
	   address calculation.  */
	.p2align 5
L(loop):
#undef RAWMEMCHR_SUBTRACT_VECTORS
#define RAWMEMCHR_SUBTRACT_VECTORS /* nothing */
	CHECK64(0,r4,pre_tail_64b)
	CHECK64(64,r4,pre_tail_64b)
	addi	  r4,r4,256

	CHECK64(0,r5,tail_64b)
	CHECK64(64,r5,tail_64b)
	addi	  r5,r5,256

	b	  L(loop)

	.p2align  5
L(pre_tail_64b):
	mr	r5,r4
L(tail_64b):
	/* OK, we found a null byte.  Let's look for it in the current 64-byte
	   block and mark it in its corresponding VR.  lxvp vx,0(ry) puts the
	   low 16B bytes into vx+1, and the high into vx, so the order here is
	   v5, v4, v7, v6.  */
	vcmpequb  v1,v5,VREG_ZERO
	vcmpequb  v2,v4,VREG_ZERO
	vcmpequb  v3,v7,VREG_ZERO
	vcmpequb  v4,v6,VREG_ZERO

	/* Take into account the other 64B blocks we had already checked.  */
	add	r5,r5,r6

	/* Extract first bit of each byte.  */
	VEXTRACTBM(r7,v1)
	VEXTRACTBM(r8,v2)
	VEXTRACTBM(r9,v3)
	VEXTRACTBM(r10,v4)

	/* Shift each value into their corresponding position.  */
	sldi	  r8,r8,16
	sldi	  r9,r9,32
	sldi	  r10,r10,48

	/* Merge the results.  */
	or	  r7,r7,r8
	or	  r8,r9,r10
	or	  r10,r8,r7

	cnttzd	  r0,r10	  /* Count trailing zeros before the match.  */
#ifndef USE_AS_RAWMEMCHR
	subf	  r5,r3,r5
#endif
	add	  r3,r5,r0	  /* Compute final length.  */
	blr

	.p2align  5
L(tail1):
	TAIL(v0,0)

	.p2align  5
L(tail2):
	TAIL(v1,16)

	.p2align  5
L(tail3):
	TAIL(v2,32)

	.p2align  5
L(tail4):
	TAIL(v3,48)

	.p2align  5
L(tail5):
	TAIL(v4,64)

	.p2align  5
L(tail6):
	TAIL(v5,80)

	.p2align  5
L(tail7):
	TAIL(v6,96)

	.p2align  5
L(tail8):
	TAIL(v7,112)

	.p2align  5
L(tail9):
	TAIL(v8,128)

	.p2align  5
L(tail10):
	TAIL(v9,144)

	.p2align  5
L(tail11):
	TAIL(v10,160)

#ifdef USE_AS_RAWMEMCHR
	.p2align  5
L(tail12):
	TAIL(v0,176)

	.p2align  5
L(tail13):
	TAIL(v1,192)

	.p2align  5
L(tail14):
	TAIL(v2,208)

	.p2align  5
L(tail15):
	TAIL(v3,224)
#endif

END (FUNCNAME)

#ifdef USE_AS_RAWMEMCHR
weak_alias (__rawmemchr,rawmemchr)
libc_hidden_builtin_def (__rawmemchr)
#else
# ifdef DEFINE_STRLEN_HIDDEN_DEF
weak_alias (__strlen, strlen)
libc_hidden_builtin_def (strlen)
# endif
#endif

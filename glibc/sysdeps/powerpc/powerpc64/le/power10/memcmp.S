/* Optimized memcmp implementation for POWER10.
   Copyright (C) 2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

/* TODO: Replace macros by the actual instructions when minimum binutils becomes
   >= 2.35.  This is used to keep compatibility with older versions.  */
#define VEXTRACTBM(rt,vrb)	 \
	.long(((4)<<(32-6))	 \
	      | ((rt)<<(32-11))  \
	      | ((8)<<(32-16))	 \
	      | ((vrb)<<(32-21)) \
	      | 1602)

#define LXVP(xtp,dq,ra)			   \
	.long(((6)<<(32-6))		   \
	      | ((((xtp)-32)>>1)<<(32-10)) \
	      | ((1)<<(32-11))		   \
	      | ((ra)<<(32-16))		   \
	      | dq)

/* Compare 32 bytes.  */
#define COMPARE_32(vr1,vr2,offset,tail_1,tail_2)\
	LXVP(32+vr1,offset,r3);		\
	LXVP(32+vr2,offset,r4);		\
	vcmpneb.  v5,vr1+1,vr2+1;	\
	bne	  cr6,L(tail_2);	\
	vcmpneb.  v4,vr1,vr2;		\
	bne	  cr6,L(tail_1);	\

#define TAIL(v_res,s1,s2)	\
	vctzlsbb  r7,v_res;	\
	vextubrx  r8,r7,s1;	\
	vextubrx  r9,r7,s2;	\
	subf	  r3,r9,r8;	\
	blr;			\

/* int [r3] memcmp (const char *s1 [r3], const char *s2 [r4],
					size_t size [r5])  */

#ifndef MEMCMP
# define MEMCMP memcmp
#endif
	.machine  power9
ENTRY_TOCLESS (MEMCMP, 4)
	CALL_MCOUNT 3

	cmpldi	cr6,r5,64
	bgt	cr6,L(loop_head)

/* Compare 64 bytes. This section is used for lengths <= 64 and for the last
   bytes for larger lengths.  */
L(last_compare):
	li	r8,16

	sldi	r9,r5,56
	sldi	r8,r8,56
	addi	r6,r3,16
	addi	r7,r4,16

	/* Align up to 16 bytes.  */
	lxvl	32+v0,r3,r9
	lxvl	32+v2,r4,r9

	/* The sub. and vcmpneb. results are concatenated by the crnand in order
	   to do a single branch. It's doing a NOT(CR0.GT AND CR6.EQ) then
	   loading to CR0.LT.  That means r9 is not bigger than 0 and v4 is not
	   all equal to 0.  */
	sub.	  r9,r9,r8
	vcmpneb.  v4,v0,v2
	crnand	  4*cr0+lt,4*cr0+gt,4*cr6+eq
	bt	  4*cr0+lt,L(tail1)

	addi	  r3,r3,32
	addi	  r4,r4,32

	lxvl	  32+v1,r6,r9
	lxvl	  32+v3,r7,r9
	sub.	  r9,r9,r8
	vcmpneb.  v5,v1,v3
	crnand	  4*cr0+lt,4*cr0+gt,4*cr6+eq
	bt	  4*cr0+lt,L(tail2)

	addi	  r6,r3,16
	addi	  r7,r4,16

	lxvl	  32+v6,r3,r9
	lxvl	  32+v8,r4,r9
	sub.	  r9,r9,r8
	vcmpneb.  v4,v6,v8
	crnand	  4*cr0+lt,4*cr0+gt,4*cr6+eq
	bt	  4*cr0+lt,L(tail3)

	lxvl	  32+v7,r6,r9
	lxvl	  32+v9,r7,r9
	vcmpneb.  v5,v7,v9
	bne	  cr6,L(tail4)

L(finish):
	/* The contents are equal.  */
	li	r3,0
	blr

L(loop_head):
	/* Calculate how many loops to run.  */
	srdi.	r8,r5,7
	beq	L(loop_tail)
	mtctr	r8

/* Main loop.  Compares 128 bytes each loop.  */
	.p2align 5
L(loop_128):
	COMPARE_32(v0,v2,0,tail1,tail2)
	COMPARE_32(v6,v8,32,tail3,tail4)
	COMPARE_32(v10,v12,64,tail5,tail6)
	COMPARE_32(v14,v16,96,tail7,tail8)

	addi	r3,r3,128
	addi	r4,r4,128
	bdnz	L(loop_128)

	/* Account loop comparisons.  */
	clrldi.  r5,r5,57
	beq	 L(finish)

/* Compares 64 bytes if length is still bigger than 64 bytes.  */
	.p2align 5
L(loop_tail):
	cmpldi	r5,64
	ble	L(last_compare)
	COMPARE_32(v0,v2,0,tail1,tail2)
	COMPARE_32(v6,v8,32,tail3,tail4)
	addi	r3,r3,64
	addi	r4,r4,64
	subi	r5,r5,64
	b	L(last_compare)

L(tail1):
	TAIL(v4,v0,v2)

L(tail2):
	TAIL(v5,v1,v3)

L(tail3):
	TAIL(v4,v6,v8)

L(tail4):
	TAIL(v5,v7,v9)

L(tail5):
	TAIL(v4,v10,v12)

L(tail6):
	TAIL(v5,v11,v13)

L(tail7):
	TAIL(v4,v14,v16)

L(tail8):
	TAIL(v5,v15,v17)

END (MEMCMP)
libc_hidden_builtin_def (memcmp)
weak_alias (memcmp, bcmp)

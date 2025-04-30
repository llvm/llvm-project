/* Shift a limb left, low level routine.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

/* mp_limb_t mpn_lshift (mp_ptr wp, mp_srcptr up, mp_size_t usize,
			 unsigned int cnt)  */

EALIGN (__mpn_lshift, 3, 0)

	mtctr	r5		# copy size into CTR
	cmplwi	cr0,r5,16	# is size < 16
	slwi	r0,r5,2
	add	r7,r3,r0	# make r7 point at end of res
	add	r4,r4,r0	# make r4 point at end of s1
	lwzu	r11,-4(r4)	# load first s1 limb
	subfic	r8,r6,32
	srw	r3,r11,r8	# compute function return value
	bge	cr0,L(big)	# branch if size >= 16

	bdz	L(end1)

L(0):	lwzu	r10,-4(r4)
	slw	r9,r11,r6
	srw	r12,r10,r8
	or	r9,r9,r12
	stwu	r9,-4(r7)
	bdz	L(end2)
	lwzu	r11,-4(r4)
	slw	r9,r10,r6
	srw	r12,r11,r8
	or	r9,r9,r12
	stwu	r9,-4(r7)
	bdnz	L(0)

L(end1):slw	r0,r11,r6
	stw	r0,-4(r7)
	blr


/* Guaranteed not to succeed.  */
L(boom): tweq    r0,r0

/* We imitate a case statement, by using (yuk!) fixed-length code chunks,
   of size 4*12 bytes.  We have to do this (or something) to make this PIC.  */
L(big):	mflr    r9
	cfi_register(lr,r9)
	bltl-   cr0,L(boom)	# Never taken, only used to set LR.
	slwi    r10,r6,4
	mflr    r12
	add     r10,r12,r10
	slwi	r8,r6,5
	add     r10,r8,r10
	mtctr   r10
	addi	r5,r5,-1
	mtlr	r9
	cfi_same_value (lr)
	bctr

L(end2):slw	r0,r10,r6
	stw	r0,-4(r7)
	blr

#define DO_LSHIFT(n) \
	mtctr	r5;							\
L(n):	lwzu	r10,-4(r4);						\
	slwi	r9,r11,n;						\
	inslwi	r9,r10,n,32-n;					\
	stwu	r9,-4(r7);						\
	bdz-	L(end2);						\
	lwzu	r11,-4(r4);						\
	slwi	r9,r10,n;						\
	inslwi	r9,r11,n,32-n;					\
	stwu	r9,-4(r7);						\
	bdnz	L(n);							\
	b	L(end1)

	DO_LSHIFT(1)
	DO_LSHIFT(2)
	DO_LSHIFT(3)
	DO_LSHIFT(4)
	DO_LSHIFT(5)
	DO_LSHIFT(6)
	DO_LSHIFT(7)
	DO_LSHIFT(8)
	DO_LSHIFT(9)
	DO_LSHIFT(10)
	DO_LSHIFT(11)
	DO_LSHIFT(12)
	DO_LSHIFT(13)
	DO_LSHIFT(14)
	DO_LSHIFT(15)
	DO_LSHIFT(16)
	DO_LSHIFT(17)
	DO_LSHIFT(18)
	DO_LSHIFT(19)
	DO_LSHIFT(20)
	DO_LSHIFT(21)
	DO_LSHIFT(22)
	DO_LSHIFT(23)
	DO_LSHIFT(24)
	DO_LSHIFT(25)
	DO_LSHIFT(26)
	DO_LSHIFT(27)
	DO_LSHIFT(28)
	DO_LSHIFT(29)
	DO_LSHIFT(30)
	DO_LSHIFT(31)

END (__mpn_lshift)

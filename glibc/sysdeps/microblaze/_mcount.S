/* Copyright (C) 1997-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

	.global	_mcount
	.type	_mcount, @function
_mcount:
	# Save regs.
	addik   r1, r1,-4 * 24
	swi	r31, r1, 4 * 23
	swi	r30, r1, 4 * 22
	swi	r29, r1, 4 * 21
	swi	r28, r1, 4 * 20
	swi	r27, r1, 4 * 19
	swi	r26, r1, 4 * 18
	swi	r25, r1, 4 * 17
	swi	r24, r1, 4 * 16
	swi	r23, r1, 4 * 15
	swi	r22, r1, 4 * 14
	swi	r21, r1, 4 * 13
	swi	r20, r1, 4 * 12
	swi	r19, r1, 4 * 11
	swi	r15, r1, 4 * 10
	swi	r12, r1, 4 * 9
	swi	r11, r1, 4 * 8
	swi	r10, r1, 4 * 7
	swi	r8, r1, 4 * 6
	swi	r7, r1, 4 * 5
	swi	r6, r1, 4 * 4
	swi	r5, r1, 4 * 3
	swi	r4, r1, 4 * 2
	swi	r3, r1, 4 * 1
	swi	r2, r1, 4 * 0

#ifdef SHARED
	mfs     r20,rpc
	addik   r20,r20,_GLOBAL_OFFSET_TABLE_+8
	addk	r6, r0, r15	# callee is in r15.
	brlid	r15, __mcount_internal@PLT
	lwi	r5, r1, 4 * 24	# caller is on the stack.
#else
	# Pass from/to in r5 and r6.
	addk	r6, r0, r15	# callee is in r15.
	bralid	r15, __mcount_internal
	lwi	r5, r1, 4 * 24	# caller is on the stack.
#endif

	# Reload regs
	lwi	r31, r1, 4 * 23
	lwi	r30, r1, 4 * 22
	lwi	r29, r1, 4 * 21
	lwi	r28, r1, 4 * 20
	lwi	r27, r1, 4 * 19
	lwi	r26, r1, 4 * 18
	lwi	r25, r1, 4 * 17
	lwi	r24, r1, 4 * 16
	lwi	r23, r1, 4 * 15
	lwi	r22, r1, 4 * 14
	lwi	r21, r1, 4 * 13
	lwi	r20, r1, 4 * 12
	lwi	r19, r1, 4 * 11
	lwi	r15, r1, 4 * 10
	lwi	r12, r1, 4 * 9
	lwi	r11, r1, 4 * 8
	lwi	r10, r1, 4 * 7
	lwi	r8, r1, 4 * 6
	lwi	r7, r1, 4 * 5
	lwi	r6, r1, 4 * 4
	lwi	r5, r1, 4 * 3
	lwi	r4, r1, 4 * 2
	lwi	r3, r1, 4 * 1
	lwi	r2, r1, 4 * 0

	rtsd	r15, 8
	addik	r1, r1, 4 * 24

	.size	_mcount, . - _mcount

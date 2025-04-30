/* Copyright (C) 2005-2021 Free Software Foundation, Inc.

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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

	.text
ENTRY (syscall)
	addk	r12,r0,r5
	addk	r5,r0,r6
	addk	r6,r0,r7
	addk	r7,r0,r8
	addk	r8,r0,r9
	addk	r9,r0,r10
	lwi	r10,r1,28
	brki	r14,8
	addk	r0,r0,r0
	addik	r4,r0,-4095
	cmpu	r4,r4,r3
	bgei	r4,SYSCALL_ERROR_LABEL
	rtsd	r15,8
	nop
PSEUDO_END (syscall)

/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

/*
    General Purpose Register (GPR) restore routine
    when Floating Point Registers (FPRs) are saved
*/

#include <sysdep.h>

ENTRY(_restgpr1_all)
		.globl	C_TEXT(_restgpr1_13)
C_TEXT(_restgpr1_13):	lwz	r13,-76(r12)
		.globl	C_TEXT(_restgpr1_14)
C_TEXT(_restgpr1_14):	lwz	r14,-72(r12)
		.globl	C_TEXT(_restgpr1_15)
C_TEXT(_restgpr1_15):	lwz	r15,-68(r12)
		.globl	C_TEXT(_restgpr1_16)
C_TEXT(_restgpr1_16):	lwz	r16,-64(r12)
		.globl	C_TEXT(_restgpr1_17)
C_TEXT(_restgpr1_17):	lwz	r17,-60(r12)
		.globl	C_TEXT(_restgpr1_18)
C_TEXT(_restgpr1_18):	lwz	r18,-56(r12)
		.globl	C_TEXT(_restgpr1_19)
C_TEXT(_restgpr1_19):	lwz	r19,-52(r12)
		.globl	C_TEXT(_restgpr1_20)
C_TEXT(_restgpr1_20):	lwz	r20,-48(r12)
		.globl	C_TEXT(_restgpr1_21)
C_TEXT(_restgpr1_21):	lwz	r21,-44(r12)
		.globl	C_TEXT(_restgpr1_22)
C_TEXT(_restgpr1_22):	lwz	r22,-40(r12)
		.globl	C_TEXT(_restgpr1_23)
C_TEXT(_restgpr1_23):	lwz	r23,-36(r12)
		.globl	C_TEXT(_restgpr1_24)
C_TEXT(_restgpr1_24):	lwz	r24,-32(r12)
		.globl	C_TEXT(_restgpr1_25)
C_TEXT(_restgpr1_25):	lwz	r25,-28(r12)
		.globl	C_TEXT(_restgpr1_26)
C_TEXT(_restgpr1_26):	lwz	r26,-24(r12)
		.globl	C_TEXT(_restgpr1_27)
C_TEXT(_restgpr1_27):	lwz	r27,-20(r12)
		.globl	C_TEXT(_restgpr1_28)
C_TEXT(_restgpr1_28):	lwz	r28,-16(r12)
		.globl	C_TEXT(_restgpr1_29)
C_TEXT(_restgpr1_29):	lwz	r29,-12(r12)	#restore r29
			lwz	r30,-8(r12)	#restore r30
			lwz	r31,-4(r12)	#restore r31
			blr			#return
END (_restgpr1_all)

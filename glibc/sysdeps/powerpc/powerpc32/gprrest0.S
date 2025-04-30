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
    when Floating Point Registers (FPRs) are not saved

    Note: This restore routine must not be called when GPR30 or
    GPR31, or both, are the only registers beings saved. In these
    cases, the saving and restoring must be done inline.
*/

#include <sysdep.h>

ENTRY(_restgpr0_all)
		.globl C_TEXT(_restgpr0_13)
C_TEXT(_restgpr0_13):	lwz	r13,-76(r1)
		.globl C_TEXT(_restgpr0_14)
C_TEXT(_restgpr0_14):	lwz	r14,-72(r1)
		.globl C_TEXT(_restgpr0_15)
C_TEXT(_restgpr0_15):	lwz	r15,-68(r1)
		.globl C_TEXT(_restgpr0_16)
C_TEXT(_restgpr0_16):	lwz	r16,-64(r1)
		.globl C_TEXT(_restgpr0_17)
C_TEXT(_restgpr0_17):	lwz	r17,-60(r1)
		.globl C_TEXT(_restgpr0_18)
C_TEXT(_restgpr0_18):	lwz	r18,-56(r1)
		.globl C_TEXT(_restgpr0_19)
C_TEXT(_restgpr0_19):	lwz	r19,-52(r1)
		.globl C_TEXT(_restgpr0_20)
C_TEXT(_restgpr0_20):	lwz	r20,-48(r1)
		.globl C_TEXT(_restgpr0_21)
C_TEXT(_restgpr0_21):	lwz	r21,-44(r1)
		.globl C_TEXT(_restgpr0_22)
C_TEXT(_restgpr0_22):	lwz	r22,-40(r1)
		.globl C_TEXT(_restgpr0_23)
C_TEXT(_restgpr0_23):	lwz	r23,-36(r1)
		.globl C_TEXT(_restgpr0_24)
C_TEXT(_restgpr0_24):	lwz	r24,-32(r1)
		.globl C_TEXT(_restgpr0_25)
C_TEXT(_restgpr0_25):	lwz	r25,-28(r1)
		.globl C_TEXT(_restgpr0_26)
C_TEXT(_restgpr0_26):	lwz	r26,-24(r1)
		.globl C_TEXT(_restgpr0_27)
C_TEXT(_restgpr0_27):	lwz	r27,-20(r1)
		.globl C_TEXT(_restgpr0_28)
C_TEXT(_restgpr0_28):	lwz	r28,-16(r1)
		.globl C_TEXT(_restgpr0_29)
C_TEXT(_restgpr0_29):	lwz	r0,8(r1)	#get return address from frame
			lwz	r29,-12(r1)	#restore r29
			mtlr	r0		#move return address to LR
			lwz	r30,-8(r1)	#restore r30
			lwz	r31,-4(r1)	#restore r31
			blr			#return
END (_restgpr0_all)

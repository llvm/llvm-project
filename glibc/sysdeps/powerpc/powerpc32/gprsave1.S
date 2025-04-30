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
    General Purpose Register (GPR) save routine
    when Floating Point Registers (FPRs) are saved
*/

#include <sysdep.h>

ENTRY(_savegpr1_all)
		.globl C_TEXT(_savegpr1_13)
C_TEXT(_savegpr1_13):	stw	r13,-76(r12)
		.globl C_TEXT(_savegpr1_14)
C_TEXT(_savegpr1_14):	stw	r14,-72(r12)
		.globl C_TEXT(_savegpr1_15)
C_TEXT(_savegpr1_15):	stw	r15,-68(r12)
		.globl C_TEXT(_savegpr1_16)
C_TEXT(_savegpr1_16):	stw	r16,-64(r12)
		.globl C_TEXT(_savegpr1_17)
C_TEXT(_savegpr1_17):	stw	r17,-60(r12)
		.globl C_TEXT(_savegpr1_18)
C_TEXT(_savegpr1_18):	stw	r18,-56(r12)
		.globl C_TEXT(_savegpr1_19)
C_TEXT(_savegpr1_19):	stw	r19,-52(r12)
		.globl C_TEXT(_savegpr1_20)
C_TEXT(_savegpr1_20):	stw	r20,-48(r12)
		.globl C_TEXT(_savegpr1_21)
C_TEXT(_savegpr1_21):	stw	r21,-44(r12)
		.globl C_TEXT(_savegpr1_22)
C_TEXT(_savegpr1_22):	stw	r22,-40(r12)
		.globl C_TEXT(_savegpr1_23)
C_TEXT(_savegpr1_23):	stw	r23,-36(r12)
		.globl C_TEXT(_savegpr1_24)
C_TEXT(_savegpr1_24):	stw	r24,-32(r12)
		.globl C_TEXT(_savegpr1_25)
C_TEXT(_savegpr1_25):	stw	r25,-28(r12)
		.globl C_TEXT(_savegpr1_26)
C_TEXT(_savegpr1_26):	stw	r26,-24(r12)
		.globl C_TEXT(_savegpr1_27)
C_TEXT(_savegpr1_27):	stw	r27,-20(r12)
		.globl C_TEXT(_savegpr1_28)
C_TEXT(_savegpr1_28):	stw	r28,-16(r12)
		.globl C_TEXT(_savegpr1_29)
C_TEXT(_savegpr1_29):	stw	r29,-12(r12)	#save r29
			stw	r30,-8(r12)	#save r30
			stw	r31,-4(r12)	#save r31
			blr			#return
END (_savegpr1_all)

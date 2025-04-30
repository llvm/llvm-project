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
    when Floating Point Registers (FPRs) are not saved

    Note: This save routine must not be called when GPR30 or
    GPR31, or both, are the only registers beings saved. In these
    cases, the saving and restoring must be done inline.
*/

#include <sysdep.h>

ENTRY(_savegpr0_all)
		.globl C_TEXT(_savegpr0_13)
C_TEXT(_savegpr0_13):	stw	r13,-76(r1)
			cfi_offset(r13,-76)
		.globl C_TEXT(_savegpr0_14)
C_TEXT(_savegpr0_14):	stw	r14,-72(r1)
			cfi_offset(r14,-72)
		.globl C_TEXT(_savegpr0_15)
C_TEXT(_savegpr0_15):	stw	r15,-68(r1)
			cfi_offset(r15,-68)
		.globl C_TEXT(_savegpr0_16)
C_TEXT(_savegpr0_16):	stw	r16,-64(r1)
			cfi_offset(r16,-64)
		.globl C_TEXT(_savegpr0_17)
C_TEXT(_savegpr0_17):	stw	r17,-60(r1)
			cfi_offset(r17,-60)
		.globl C_TEXT(_savegpr0_18)
C_TEXT(_savegpr0_18):	stw	r18,-56(r1)
			cfi_offset(r18,-56)
		.globl C_TEXT(_savegpr0_19)
C_TEXT(_savegpr0_19):	stw	r19,-52(r1)
			cfi_offset(r19,-52)
		.globl C_TEXT(_savegpr0_20)
C_TEXT(_savegpr0_20):	stw	r20,-48(r1)
			cfi_offset(r20,-48)
		.globl C_TEXT(_savegpr0_21)
C_TEXT(_savegpr0_21):	stw	r21,-44(r1)
			cfi_offset(r21,-44)
		.globl C_TEXT(_savegpr0_22)
C_TEXT(_savegpr0_22):	stw	r22,-40(r1)
			cfi_offset(r22,-40)
		.globl C_TEXT(_savegpr0_23)
C_TEXT(_savegpr0_23):	stw	r23,-36(r1)
			cfi_offset(r23,-36)
		.globl C_TEXT(_savegpr0_24)
C_TEXT(_savegpr0_24):	stw	r24,-32(r1)
			cfi_offset(r24,-32)
		.globl C_TEXT(_savegpr0_25)
C_TEXT(_savegpr0_25):	stw	r25,-28(r1)
			cfi_offset(r25,-28)
		.globl C_TEXT(_savegpr0_26)
C_TEXT(_savegpr0_26):	stw	r26,-24(r1)
			cfi_offset(r26,-24)
		.globl C_TEXT(_savegpr0_27)
C_TEXT(_savegpr0_27):	stw	r27,-20(r1)
			cfi_offset(r27,-20)
		.globl C_TEXT(_savegpr0_28)
C_TEXT(_savegpr0_28):	stw	r28,-16(r1)
			cfi_offset(r28,-16)
		.globl C_TEXT(_savegpr0_29)
C_TEXT(_savegpr0_29):	stw	r29,-12(r1) #save r29
			stw	r30,-8(r1)  #save r30
			stw	r31,-4(r1)  #save r31
			cfi_offset(r29,-12)
			cfi_offset(r30,-8)
			cfi_offset(r31,-4)
			stw	r0,8(r1)    #save LR in callers frame
			blr		    #return
END (_savegpr0_all)

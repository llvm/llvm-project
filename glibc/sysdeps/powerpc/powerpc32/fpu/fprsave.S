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
    Floating Point Registers (FPRs) save routine
*/

#include <sysdep.h>

ENTRY(_savefpr_all)
                .globl C_TEXT(_savef14)
		.globl C_TEXT(_savefpr_14)
C_TEXT(_savef14):
C_TEXT(_savefpr_14):	stfd	fp14,-144(r1)
			cfi_offset(fp14,-144)
		.globl C_TEXT(_savef15)
		.globl C_TEXT(_savefpr_15)
C_TEXT(_savef15):
C_TEXT(_savefpr_15):	stfd	fp15,-136(r1)
			cfi_offset(fp15,-136)
		.globl C_TEXT(_savef16)
		.globl C_TEXT(_savefpr_16)
C_TEXT(_savef16):
C_TEXT(_savefpr_16):	stfd	fp16,-128(r1)
			cfi_offset(fp16,-128)
		.globl C_TEXT(_savef17)
		.globl C_TEXT(_savefpr_17)
C_TEXT(_savef17):
C_TEXT(_savefpr_17):	stfd	fp17,-120(r1)
			cfi_offset(fp17,-120)
		.globl C_TEXT(_savef18)
		.globl C_TEXT(_savefpr_18)
C_TEXT(_savef18):
C_TEXT(_savefpr_18):	stfd	fp18,-112(r1)
			cfi_offset(fp18,-112)
		.globl C_TEXT(_savef19)
		.globl C_TEXT(_savefpr_19)
C_TEXT(_savef19):
C_TEXT(_savefpr_19):	stfd	fp19,-104(r1)
			cfi_offset(fp19,-104)
		.globl C_TEXT(_savef20)
		.globl C_TEXT(_savefpr_20)
C_TEXT(_savef20):
C_TEXT(_savefpr_20):	stfd	fp20,-96(r1)
			cfi_offset(fp20,-96)
		.globl C_TEXT(_savef21)
		.globl C_TEXT(_savefpr_21)
C_TEXT(_savef21):
C_TEXT(_savefpr_21):	stfd	fp21,-88(r1)
			cfi_offset(fp21,-88)
		.globl C_TEXT(_savef22)
		.globl C_TEXT(_savefpr_22)
C_TEXT(_savef22):
C_TEXT(_savefpr_22):	stfd	fp22,-80(r1)
			cfi_offset(fp22,-80)
		.globl C_TEXT(_savef23)
		.globl C_TEXT(_savefpr_23)
C_TEXT(_savef23):
C_TEXT(_savefpr_23):	stfd	fp23,-72(r1)
			cfi_offset(fp23,-72)
		.globl C_TEXT(_savef24)
		.globl C_TEXT(_savefpr_24)
C_TEXT(_savef24):
C_TEXT(_savefpr_24):	stfd	fp24,-64(r1)
			cfi_offset(fp24,-64)
		.globl C_TEXT(_savef25)
		.globl C_TEXT(_savefpr_25)
C_TEXT(_savef25):
C_TEXT(_savefpr_25):	stfd	fp25,-56(r1)
			cfi_offset(fp25,-56)
		.globl C_TEXT(_savef26)
		.globl C_TEXT(_savefpr_26)
C_TEXT(_savef26):
C_TEXT(_savefpr_26):	stfd	fp26,-48(r1)
			cfi_offset(fp26,-48)
		.globl C_TEXT(_savef27)
		.globl C_TEXT(_savefpr_27)
C_TEXT(_savef27):
C_TEXT(_savefpr_27):	stfd	fp27,-40(r1)
			cfi_offset(fp27,-40)
		.globl C_TEXT(_savef28)
		.globl C_TEXT(_savefpr_28)
C_TEXT(_savef28):
C_TEXT(_savefpr_28):	stfd	fp28,-32(r1)
			cfi_offset(fp28,-32)
		.globl C_TEXT(_savef29)
		.globl C_TEXT(_savefpr_29)
C_TEXT(_savef29):
C_TEXT(_savefpr_29):	stfd	fp29,-24(r1)	#save f29
			stfd	fp30,-16(r1)	#save f30
			stfd	fp31,-8(r1)	#save f31
			cfi_offset(fp29,-24)
			cfi_offset(fp30,-16)
			cfi_offset(fp31,-8)
			stw	r0,8(r1)	#save LR in callers frame
			blr			#return
END (_savefpr_all)

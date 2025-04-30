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
    Floating Point Registers (FPRs) restore routine
*/

#include <sysdep.h>

ENTRY(_restfpr_all)
		.globl C_TEXT(_restf14)
		.globl C_TEXT(_restfpr_14)
C_TEXT(_restf14):
C_TEXT(_restfpr_14):	lfd	fp14,-144(r1)
		.globl C_TEXT(_restf15)
		.globl C_TEXT(_restfpr_15)
C_TEXT(_restf15):
C_TEXT(_restfpr_15):	lfd	fp15,-136(r1)
		.globl C_TEXT(_restf16)
		.globl C_TEXT(_restfpr_16)
C_TEXT(_restf16):
C_TEXT(_restfpr_16):	lfd	fp16,-128(r1)
		.globl C_TEXT(_restf17)
		.globl C_TEXT(_restfpr_17)
C_TEXT(_restf17):
C_TEXT(_restfpr_17):	lfd	fp17,-120(r1)
		.globl C_TEXT(_restf18)
		.globl C_TEXT(_restfpr_18)
C_TEXT(_restf18):
C_TEXT(_restfpr_18):	lfd	fp18,-112(r1)
		.globl C_TEXT(_restf19)
		.globl C_TEXT(_restfpr_19)
C_TEXT(_restf19):
C_TEXT(_restfpr_19):	lfd	fp19,-104(r1)
		.globl C_TEXT(_restf20)
		.globl C_TEXT(_restfpr_20)
C_TEXT(_restf20):
C_TEXT(_restfpr_20):	lfd	fp20,-96(r1)
		.globl C_TEXT(_restf21)
		.globl C_TEXT(_restfpr_21)
C_TEXT(_restf21):
C_TEXT(_restfpr_21):	lfd	fp21,-88(r1)
		.globl C_TEXT(_restf22)
		.globl C_TEXT(_restfpr_22)
C_TEXT(_restf22):
C_TEXT(_restfpr_22):	lfd	fp22,-80(r1)
		.globl C_TEXT(_restf23)
		.globl C_TEXT(_restfpr_23)
C_TEXT(_restf23):
C_TEXT(_restfpr_23):	lfd	fp23,-72(r1)
		.globl C_TEXT(_restf24)
		.globl C_TEXT(_restfpr_24)
C_TEXT(_restf24):
C_TEXT(_restfpr_24):	lfd	fp24,-64(r1)
		.globl C_TEXT(_restf25)
		.globl C_TEXT(_restfpr_25)
C_TEXT(_restf25):
C_TEXT(_restfpr_25):	lfd	fp25,-56(r1)
		.globl C_TEXT(_restf26)
		.globl C_TEXT(_restfpr_26)
C_TEXT(_restf26):
C_TEXT(_restfpr_26):	lfd	fp26,-48(r1)
		.globl C_TEXT(_restf27)
		.globl C_TEXT(_restfpr_27)
C_TEXT(_restf27):
C_TEXT(_restfpr_27):	lfd	fp27,-40(r1)
		.globl C_TEXT(_restf28)
		.globl C_TEXT(_restfpr_28)
C_TEXT(_restf28):
C_TEXT(_restfpr_28):	lfd	fp28,-32(r1)
		.globl C_TEXT(_restf29)
		.globl C_TEXT(_restfpr_29)
C_TEXT(_restf29):
C_TEXT(_restfpr_29):	lwz	r0,8(r1)	#get return address from frame
			lfd	fp29,-24(r1)    #restore f29
			mtlr	r0		#move return address to LR
			lfd	fp30,-16(r1)	#restore f30
			lfd	fp31,-8(r1)	#restore f31
			blr			#return
END (_restfpr_all)

/* mc68020 __mpn_addmul_1 -- Multiply a limb vector with a limb and add
   the result to a second limb vector.

Copyright (C) 1992-2021 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or (at your
option) any later version.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MP Library.  If not, see <https://www.gnu.org/licenses/>.  */

/*
  INPUT PARAMETERS
  res_ptr	(sp + 4)
  s1_ptr	(sp + 8)
  s1_size	(sp + 12)
  s2_limb	(sp + 16)
*/

#include "sysdep.h"
#include "asm-syntax.h"

	TEXT
ENTRY(__mpn_addmul_1)

#define res_ptr a0
#define s1_ptr a1
#define s1_size d2
#define s2_limb d4

/* Save used registers on the stack.  */
	moveml	R(d2)-R(d5),MEM_PREDEC(sp)
	cfi_adjust_cfa_offset (4*4)
	cfi_rel_offset (R(d2), 0)
	cfi_rel_offset (R(d3), 4)
	cfi_rel_offset (R(d4), 8)
	cfi_rel_offset (R(d5), 12)

/* Copy the arguments to registers.  Better use movem?  */
	movel	MEM_DISP(sp,20),R(res_ptr)
	movel	MEM_DISP(sp,24),R(s1_ptr)
	movel	MEM_DISP(sp,28),R(s1_size)
	movel	MEM_DISP(sp,32),R(s2_limb)

	eorw	#1,R(s1_size)
	clrl	R(d1)
	clrl	R(d5)
	lsrl	#1,R(s1_size)
	bcc	L(L1)
	subql	#1,R(s1_size)
	subl	R(d0),R(d0)		/* (d0,cy) <= (0,0) */

L(Loop:)
	movel	MEM_POSTINC(s1_ptr),R(d3)
	mulul	R(s2_limb),R(d1):R(d3)
	addxl	R(d0),R(d3)
	addxl	R(d5),R(d1)
	addl	R(d3),MEM_POSTINC(res_ptr)
L(L1:)	movel	MEM_POSTINC(s1_ptr),R(d3)
	mulul	R(s2_limb),R(d0):R(d3)
	addxl	R(d1),R(d3)
	addxl	R(d5),R(d0)
	addl	R(d3),MEM_POSTINC(res_ptr)

	dbf	R(s1_size),L(Loop)
	addxl	R(d5),R(d0)
	subl	#0x10000,R(s1_size)
	bcc	L(Loop)

/* Restore used registers from stack frame.  */
	moveml	MEM_POSTINC(sp),R(d2)-R(d5)
	cfi_adjust_cfa_offset (-4*4)
	cfi_restore (R(d2))
	cfi_restore (R(d3))
	cfi_restore (R(d4))
	cfi_restore (R(d5))

	rts
END(__mpn_addmul_1)

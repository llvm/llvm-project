/* mc68020 __mpn_rshift -- Shift right a low-level natural-number integer.

Copyright (C) 1996-2021 Free Software Foundation, Inc.

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
  s_ptr		(sp + 8)
  s_size	(sp + 16)
  cnt		(sp + 12)
*/

#include "sysdep.h"
#include "asm-syntax.h"

#define res_ptr a1
#define s_ptr a0
#define s_size d6
#define cnt d4

	TEXT
ENTRY(__mpn_rshift)
/* Save used registers on the stack.  */
	moveml	R(d2)-R(d6)/R(a2),MEM_PREDEC(sp)
	cfi_adjust_cfa_offset (6*4)
	cfi_rel_offset (R(d2), 0)
	cfi_rel_offset (R(d3), 4)
	cfi_rel_offset (R(d4), 8)
	cfi_rel_offset (R(d5), 12)
	cfi_rel_offset (R(d6), 16)
	cfi_rel_offset (R(a2), 20)

/* Copy the arguments to registers.  */
	movel	MEM_DISP(sp,28),R(res_ptr)
	movel	MEM_DISP(sp,32),R(s_ptr)
	movel	MEM_DISP(sp,36),R(s_size)
	movel	MEM_DISP(sp,40),R(cnt)

	moveql	#1,R(d5)
	cmpl	R(d5),R(cnt)
	bne	L(Lnormal)
	cmpl	R(res_ptr),R(s_ptr)
	bls	L(Lspecial)		/* jump if res_ptr >= s_ptr */
#if (defined (__mc68020__) || defined (__NeXT__) || defined(mc68020))
	lea	MEM_INDX1(res_ptr,s_size,l,4),R(a2)
#else /* not mc68020 */
	movel	R(s_size),R(d0)
	asll	#2,R(d0)
	lea	MEM_INDX(res_ptr,d0,l),R(a2)
#endif
	cmpl	R(s_ptr),R(a2)
	bls	L(Lspecial)		/* jump if s_ptr >= res_ptr + s_size */

L(Lnormal:)
	moveql	#32,R(d5)
	subl	R(cnt),R(d5)
	movel	MEM_POSTINC(s_ptr),R(d2)
	movel	R(d2),R(d0)
	lsll	R(d5),R(d0)		/* compute carry limb */

	lsrl	R(cnt),R(d2)
	movel	R(d2),R(d1)
	subql	#1,R(s_size)
	beq	L(Lend)
	lsrl	#1,R(s_size)
	bcs	L(L1)
	subql	#1,R(s_size)

L(Loop:)
	movel	MEM_POSTINC(s_ptr),R(d2)
	movel	R(d2),R(d3)
	lsll	R(d5),R(d3)
	orl	R(d3),R(d1)
	movel	R(d1),MEM_POSTINC(res_ptr)
	lsrl	R(cnt),R(d2)
L(L1:)
	movel	MEM_POSTINC(s_ptr),R(d1)
	movel	R(d1),R(d3)
	lsll	R(d5),R(d3)
	orl	R(d3),R(d2)
	movel	R(d2),MEM_POSTINC(res_ptr)
	lsrl	R(cnt),R(d1)

	dbf	R(s_size),L(Loop)
	subl	#0x10000,R(s_size)
	bcc	L(Loop)

L(Lend:)
	movel	R(d1),MEM(res_ptr) /* store most significant limb */

/* Restore used registers from stack frame.  */
	moveml	MEM_POSTINC(sp),R(d2)-R(d6)/R(a2)
	cfi_remember_state
	cfi_adjust_cfa_offset (-6*4)
	cfi_restore (R(d2))
	cfi_restore (R(d3))
	cfi_restore (R(d4))
	cfi_restore (R(d5))
	cfi_restore (R(d6))
	cfi_restore (R(a2))
	rts

/* We loop from most significant end of the arrays, which is only
   permissible if the source and destination don't overlap, since the
   function is documented to work for overlapping source and destination.  */

	cfi_restore_state
L(Lspecial:)
#if (defined (__mc68020__) || defined (__NeXT__) || defined(mc68020))
	lea	MEM_INDX1(s_ptr,s_size,l,4),R(s_ptr)
	lea	MEM_INDX1(res_ptr,s_size,l,4),R(res_ptr)
#else /* not mc68000 */
	movel	R(s_size),R(d0)
	asll	#2,R(d0)
	addl	R(s_size),R(s_ptr)
	addl	R(s_size),R(res_ptr)
#endif

	clrl	R(d0)			/* initialize carry */
	eorw	#1,R(s_size)
	lsrl	#1,R(s_size)
	bcc	L(LL1)
	subql	#1,R(s_size)

L(LLoop:)
	movel	MEM_PREDEC(s_ptr),R(d2)
	roxrl	#1,R(d2)
	movel	R(d2),MEM_PREDEC(res_ptr)
L(LL1:)
	movel	MEM_PREDEC(s_ptr),R(d2)
	roxrl	#1,R(d2)
	movel	R(d2),MEM_PREDEC(res_ptr)

	dbf	R(s_size),L(LLoop)
	roxrl	#1,R(d0)		/* save cy in msb */
	subl	#0x10000,R(s_size)
	bcs	L(LLend)
	addl	R(d0),R(d0)		/* restore cy */
	bra	L(LLoop)

L(LLend:)
/* Restore used registers from stack frame.  */
	moveml	MEM_POSTINC(sp),R(d2)-R(d6)/R(a2)
	cfi_adjust_cfa_offset (-6*4)
	cfi_restore (R(d2))
	cfi_restore (R(d3))
	cfi_restore (R(d4))
	cfi_restore (R(d5))
	cfi_restore (R(d6))
	cfi_restore (R(a2))
	rts
END(__mpn_rshift)

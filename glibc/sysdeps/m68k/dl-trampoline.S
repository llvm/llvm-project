/* PLT trampolines.  m68k version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#if !defined (__mcoldfire__)
# define FMOVE		fmove.x
# define FPSPACE	12
#elif defined (__mcffpu__)
# define FMOVE		fmove.d
# define FPSPACE	8
#else
# define FPSPACE	0
#endif

	.text
	.globl _dl_runtime_resolve
	.type _dl_runtime_resolve, @function
_dl_runtime_resolve:
	cfi_startproc
	cfi_adjust_cfa_offset (8)
	| Save %a0 (struct return address) and %a1.
	move.l %a0, -(%sp)
	cfi_adjust_cfa_offset (4)
	move.l %a1, -(%sp)
	cfi_adjust_cfa_offset (4)
	| Call the real address resolver.
	jbsr _dl_fixup
	| Restore register %a0 and %a1.
	move.l (%sp)+, %a1
	cfi_adjust_cfa_offset (-4)
	move.l (%sp)+, %a0
	cfi_adjust_cfa_offset (-4)
	| Pop parameters
	addq.l #8, %sp
	cfi_adjust_cfa_offset (-8)
	| Call real function.
#ifdef __mcoldfire__
	move.l %d0,-(%sp)
	cfi_adjust_cfa_offset (4)
	rts
#else
	jmp (%d0)
#endif
	cfi_endproc
	.size _dl_runtime_resolve, . - _dl_runtime_resolve

	.text
	.globl _dl_runtime_profile
	.type _dl_runtime_profile, @function
_dl_runtime_profile:
	cfi_startproc
	cfi_adjust_cfa_offset (8)
	pea 8(%sp)
	cfi_adjust_cfa_offset (4)
	move.l %a1, -(%sp)
	cfi_adjust_cfa_offset (4)
	move.l %a0, -(%sp)
	cfi_adjust_cfa_offset (4)
	pea -1.w
	cfi_adjust_cfa_offset (4)
	| Push parameters for _dl_profile_fixup
	pea (%sp)
	cfi_adjust_cfa_offset (4)
	pea 8(%sp)
	cfi_adjust_cfa_offset (4)
	move.l 32(%sp), -(%sp)
	cfi_adjust_cfa_offset (4)
	move.l 32(%sp), -(%sp)
	cfi_adjust_cfa_offset (4)
	move.l 32(%sp), -(%sp)
	cfi_adjust_cfa_offset (4)
	subq.l #8, %sp
	cfi_adjust_cfa_offset (8)
	| Call the real address resolver.
	jbsr _dl_profile_fixup
	| Pop parameters
	lea 28(%sp), %sp
	cfi_adjust_cfa_offset (-28)
	move.l (%sp), %d1
	jpl 1f
	addq.l #4, %sp
	cfi_adjust_cfa_offset (-4)
	| Restore register %a0 and %a1.
	move.l (%sp)+, %a0
	cfi_adjust_cfa_offset (-4)
	move.l (%sp)+, %a1
	cfi_adjust_cfa_offset (-4)
	lea 12(%sp), %sp
	cfi_adjust_cfa_offset (-12)
	| Call real function.
#ifdef __mcoldfire__
	move.l %d0,-(%sp)
	cfi_adjust_cfa_offset (4)
	rts
#else
	jmp (%d0)
#endif

	/*
	    +24     return address
	    +20     PLT1
	    +16     PLT2
	    +12     %sp
	    +8      %a1
	    +4      %a0
	   %sp      free
	*/
#ifdef __mcoldfire__
	cfi_adjust_cfa_offset (20)
#else
	cfi_adjust_cfa_offset (24)
#endif
1:	move.l %a2, (%sp)
	cfi_rel_offset (%a2, 0)
	move.l %sp, %a2
	move.l %sp, %a0
	lea 28(%sp), %a1
	| Round framesize up to longword alignment
	addq.l #3, %d1
	and.l #-3, %d1
	sub.l %d1, %a0
	move.l %a0, %sp
	cfi_def_cfa_register (%a2)
#ifdef __mcoldfire__
	tst.l %d1
	beq 2f
1:	move.l (%a1)+, (%a0)+
	subq.l #4,%d1
	bne 1b
2:
#else
	lsr.l #2,%d1
	jra 2f
1:	move.l (%a1)+, (%a0)+
2:	dbra %d1,1b
#endif
	/*
	   %a2+24  return address
	   %a2+20  PLT1
	   %a2+16  PLT2
	   %a2+12  %sp
	   %a2+8   %a1
	   %a2+4   %a0
	   %a2     %a2
	   %sp     copied stack frame
	*/

	move.l 4(%a2), %a0
	move.l 8(%a2), %a1
#ifdef __mcoldfire__
	pea 2f(%pc)
	move.l %d0,-(%sp)
	rts
2:
#else
	jsr (%d0)
#endif
	move.l %a2, %sp
	cfi_def_cfa_register (%sp)
	move.l (%sp)+, %a2
	cfi_adjust_cfa_offset (4)
	cfi_restore (%a2)
	/*
	    +20     return address
	    +16     PLT1
	    +12     PLT2
	    +8      %sp
	    +4      %a1
	   %sp      %a0
	*/
#ifdef FMOVE
	FMOVE %fp0, -(%sp)
	cfi_adjust_cfa_offset (FPSPACE)
#endif
	move.l %a0, -(%sp)
	cfi_adjust_cfa_offset (4)
	move.l %d1, -(%sp)
	cfi_adjust_cfa_offset (4)
	move.l %d0, -(%sp)
	cfi_adjust_cfa_offset (4)
	pea (%sp)
	cfi_adjust_cfa_offset (4)
	pea (16+FPSPACE)(%sp)
	cfi_adjust_cfa_offset (4)
	move.l (32+FPSPACE)(%sp), -(%sp)
	cfi_adjust_cfa_offset (4)
	move.l (32+FPSPACE)(%sp), -(%sp)
	cfi_adjust_cfa_offset (4)
	jbsr _dl_call_pltexit
	lea 16(%sp), %sp
	cfi_adjust_cfa_offset (-16)
	move.l (%sp)+, %d0
	cfi_adjust_cfa_offset (-4)
	move.l (%sp)+, %d1
	cfi_adjust_cfa_offset (-4)
	move.l (%sp)+, %a0
	cfi_adjust_cfa_offset (-4)
#ifdef FMOVE
	FMOVE (%sp)+, %fp0
	cfi_adjust_cfa_offset (-FPSPACE)
#endif
	lea 20(%sp), %sp
	cfi_adjust_cfa_offset (-20)
	rts
	cfi_endproc
	.size _dl_runtime_profile, . - _dl_runtime_profile

/* Save current context.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>

#include "ucontext_i.h"


ENTRY(__getcontext)
	/* Load address of the context data structure.  */
	move.l	4(%sp), %a0

	/* Save the preserved registers and the return address.  */
	movem.l	%d2-%d7, oGREGS+2*4(%a0)
	movem.l	%a2-%a6, oGREGS+10*4(%a0)
	lea	4(%sp), %a1
	move.l	%a1, oSP(%a0)
	move.l	(%sp), oPC(%a0)
	fmovem.l %fpsr/%fpcr/%fpiar, oFPREGS(%a0)
	fmovem.x %fp2-%fp7, oFP2(%a0)
	cfi_def_cfa (%a0, 0)
	cfi_offset (%d2, oGREGS+2*4)
	cfi_offset (%d3, oGREGS+3*4)
	cfi_offset (%pc, oPC)

	/* Save the current signal mask.  */
	lea	oSIGMASK(%a0), %a1
	move.l	%a1, %d3
	clr.l	%d2
	move.l	#SIG_SETMASK, %d1
	move.l	#__NR_sigprocmask, %d0
	trap	#0
	/* Restore registers.  */
	move.l	oGREGS+2*4(%a0), %d2
	cfi_restore (%d2)
	move.l	oGREGS+3*4(%a0), %d3
	cfi_restore (%d3)
	cfi_def_cfa (%sp, 4)
	cfi_offset (%pc, -4)
	cmp.l	#-4095, %d0		/* Check %d0 for error.  */
	jcc	SYSCALL_ERROR_LABEL	/* Jump to error handler.  */

0:	/* All done, return 0 for success.  */
	clr.l	%d0
	rts
PSEUDO_END(__getcontext)

weak_alias (__getcontext, getcontext)

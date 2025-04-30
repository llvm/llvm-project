/* Save current context and install the given one.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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
#include <ucontext-offsets.h>

ENTRY(__swapcontext)

#ifdef PROF
	ldgp	$29, 0($27)
	.set noat
	lda     AT, _mcount
	jsr     AT, (AT), _mcount
	.set at
	.prologue 1
#elif defined PIC
	.prologue 0
#else
	ldgp	$29, 0($27)
	.prologue 1
#endif

#ifdef PIC
	unop
	bsr	$0, __getcontext_x	!samegp
	mov	$17, $16
	br	$31, __setcontext	!samegp
#else
	jsr	$0, __getcontext_x
	mov	$17, $16
	jmp	$31, __setcontext
#endif

END(__swapcontext)
weak_alias (__swapcontext, swapcontext)

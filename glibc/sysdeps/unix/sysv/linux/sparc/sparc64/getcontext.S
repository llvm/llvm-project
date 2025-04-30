/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson (rth@tamu.edu).

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

/* int getcontext(ucontext_t *); */

ENTRY(__getcontext)

	ldx	[%o0 + UC_LINK], %o1	/* Preserve uc_link field, the
					   trap clears it.  */
	ta	0x6e
1:
	ldx	[%o0 + UC_M_PC], %o2
	ldx	[%o0 + UC_M_NPC], %o3
	ldx	[%o0 + __UC_SIGMASK], %o4
	stx	%o1, [%o0 + UC_LINK]
	add	%o2, 2f - 1b, %o2
	stx	%o2, [%o0 + UC_M_PC]
	add	%o3, 2f - 1b, %o3
	stx	%o3, [%o0 + UC_M_NPC]
#if SIGMASK_WORDS == 16
	stx	%o4, [%o0 + UC_SIGMASK]
	stx	%g0, [%o0 + UC_SIGMASK + 8]
	stx	%g0, [%o0 + UC_SIGMASK + 16]
	stx	%g0, [%o0 + UC_SIGMASK + 24]
	stx	%g0, [%o0 + UC_SIGMASK + 32]
	stx	%g0, [%o0 + UC_SIGMASK + 40]
	stx	%g0, [%o0 + UC_SIGMASK + 48]
	stx	%g0, [%o0 + UC_SIGMASK + 56]
	stx	%g0, [%o0 + UC_SIGMASK + 64]
	stx	%g0, [%o0 + UC_SIGMASK + 72]
	stx	%g0, [%o0 + UC_SIGMASK + 80]
	stx	%g0, [%o0 + UC_SIGMASK + 88]
	stx	%g0, [%o0 + UC_SIGMASK + 96]
	stx	%g0, [%o0 + UC_SIGMASK + 104]
	stx	%g0, [%o0 + UC_SIGMASK + 112]
	stx	%g0, [%o0 + UC_SIGMASK + 120]
#else
# error Adjust __getcontext
#endif
2:
	retl
	 clr	%o0

END(__getcontext)

weak_alias (__getcontext, getcontext)

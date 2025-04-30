/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

ENTRY(htonl)
#ifdef PROF
	ldgp	gp, 0(pv)
	.set noat
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.set at
	.prologue 1
#else
	.prologue 0
#endif

	inslh	a0, 7, t0	# t0 = 0000000000AABBCC
	inswl	a0, 3, t1	# t1 = 000000CCDD000000
	or	t1, t0, t1	# t1 = 000000CCDDAABBCC
	srl	t1, 16, t2	# t2 = 0000000000CCDDAA
	zapnot	t1, 0x0A, t0	# t0 = 00000000DD00BB00
	zapnot	t2, 0x05, t3	# t3 = 0000000000CC00AA
	addl	t0, t3, v0	# v0 = ssssssssDDCCBBAA
	ret

	END(htonl)

weak_alias (htonl, ntohl)

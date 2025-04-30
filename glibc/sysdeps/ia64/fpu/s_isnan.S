/* Test for NaN
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jes Sorensen <jes@linuxcare.com>, October 2000.

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
 * isnan(x) returns 1 is x is nan, else 0;
 */

#include <sysdep.h>
#undef ret

ENTRY (__isnan)
{
.mfi
	fclass.m p6, p7 = farg0, @nat | @qnan | @snan
	;;
}
{
.mib
(p6)	mov ret0 = 1
(p7)	mov ret0 = 0
	br.ret.sptk.many rp
}
END (__isnan)

strong_alias (__isnan, __isnanf)
strong_alias (__isnan, __isnanl)

weak_alias (__isnan, isnan)
weak_alias (__isnanf, isnanf)
weak_alias (__isnanl, isnanl)

hidden_def (__isnan)
hidden_def (__isnanf)
hidden_def (__isnanl)

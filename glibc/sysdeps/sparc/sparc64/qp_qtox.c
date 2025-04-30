/* Software floating-point emulation.
   Return (long)(*a)
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson (rth@cygnus.com) and
		  Jakub Jelinek (jj@ultra.linux.cz).

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

#define FP_ROUNDMODE FP_RND_ZERO
#include "soft-fp.h"
#include "quad.h"

long _Qp_qtox(const long double *a)
{
  FP_DECL_EX;
  FP_DECL_Q(A);
  unsigned long r;

  FP_INIT_ROUNDMODE;
  FP_UNPACK_RAW_QP(A, a);
  FP_TO_INT_Q(r, A, 64, 1);
  QP_HANDLE_EXCEPTIONS(
	long rx;
	__asm (
"	ldd [%1], %%f52\n"
"	ldd [%1+8], %%f54\n"
"	fqtox %%f52, %%f60\n"
"	std %%f60, [%0]\n"
"	" : : "r" (&rx), "r" (a) : QP_CLOBBER);
	r = rx);

  return r;
}

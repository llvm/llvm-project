/* Manipulation of the bit representation of 'long double' quantities.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#ifndef _MATH_LDBL_H_IBM128_COMPAT
#define _MATH_LDBL_H_IBM128_COMPAT 1

#include <bits/floatn.h>

/* Trampoline in the ldbl-128ibm headers if building against the
   old abi.  Otherwise, we have nothing to add. */
#if __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI == 0
#include_next <math_ldbl.h>
#endif

#endif /* _MATH_LDBL_H_IBM128_COMPAT */

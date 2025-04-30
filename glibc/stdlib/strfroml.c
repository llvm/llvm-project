/* Definitions for strfroml.  Implementation in stdlib/strfrom-skeleton.c.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <bits/floatn.h>

#define FLOAT		long double
#define STRFROM		strfroml

#if __HAVE_FLOAT128 && !__HAVE_DISTINCT_FLOAT128
# define strfromf128 __hide_strfromf128
#endif
#if __HAVE_FLOAT64X_LONG_DOUBLE
# define strfromf64x __hide_strfromf64x
#endif

#include <stdlib.h>

#if __HAVE_FLOAT128 && !__HAVE_DISTINCT_FLOAT128
# undef strfromf128
#endif
#if __HAVE_FLOAT64X_LONG_DOUBLE
# undef strfromf64x
#endif

#include "strfrom-skeleton.c"

#if __HAVE_FLOAT128 && !__HAVE_DISTINCT_FLOAT128
weak_alias (strfroml, strfromf128)
#endif
#if __HAVE_FLOAT64X_LONG_DOUBLE
weak_alias (strfroml, strfromf64x)
#endif

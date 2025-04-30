/* Definitions for strfromd.  Implementation in stdlib/strfrom-skeleton.c.
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

#define FLOAT		double
#define STRFROM		strfromd

#if __HAVE_FLOAT64 && !__HAVE_DISTINCT_FLOAT64
# define strfromf64 __hide_strfromf64
#endif
#if __HAVE_FLOAT32X && !__HAVE_DISTINCT_FLOAT32X
# define strfromf32x __hide_strfromf32x
#endif

#include <stdlib.h>

#if __HAVE_FLOAT64 && !__HAVE_DISTINCT_FLOAT64
# undef strfromf64
#endif
#if __HAVE_FLOAT32X && !__HAVE_DISTINCT_FLOAT32X
# undef strfromf32x
#endif

#include "strfrom-skeleton.c"

#if __HAVE_FLOAT64 && !__HAVE_DISTINCT_FLOAT64
weak_alias (strfromd, strfromf64)
#endif
#if __HAVE_FLOAT32X && !__HAVE_DISTINCT_FLOAT32X
weak_alias (strfromd, strfromf32x)
#endif

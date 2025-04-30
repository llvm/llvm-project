/* Definitions for strfromf.  Implementation in stdlib/strfrom-skeleton.c.
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

#define FLOAT		float
#define STRFROM		strfromf

#if __HAVE_FLOAT32 && !__HAVE_DISTINCT_FLOAT32
# define strfromf32 __hide_strfromf32
#endif

#include <stdlib.h>

#if __HAVE_FLOAT32 && !__HAVE_DISTINCT_FLOAT32
# undef strfromf32
#endif

#include "strfrom-skeleton.c"

#if __HAVE_FLOAT32 && !__HAVE_DISTINCT_FLOAT32
weak_alias (strfromf, strfromf32)
#endif

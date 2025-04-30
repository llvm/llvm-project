/* wide character shim for tst-strtod-round-skeleton.c.
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

#include <wchar.h>

/* Include stdio.h early to avoid issues with the snprintf
   redefinition below.  */
#include <stdio.h>

#define L_(str) L ## str
#define FNPFX wcs
#define CHAR wchar_t
#define STRM "%ls"
#define snprintf swprintf

#include <stdlib/tst-strtod-round-skeleton.c>

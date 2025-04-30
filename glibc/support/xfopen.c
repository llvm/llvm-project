/* fopen with error checking.
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

#include <support/xstdio.h>

#include <support/check.h>
#include <stdlib.h>

FILE *
xfopen (const char *path, const char *mode)
{
  FILE *fp = fopen (path, mode);
  if (fp == NULL)
    FAIL_EXIT1 ("could not open %s (mode \"%s\"): %m", path, mode);
  return fp;
}

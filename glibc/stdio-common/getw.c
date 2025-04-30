/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <stdio.h>

#include <libio/iolibio.h>
#define fread(p, m, n, s) _IO_fread (p, m, n, s)

/* Read a word (int) from STREAM.  */
int
getw (FILE *stream)
{
  int w;

  /* Is there a better way?  */
  if (fread ((void *) &w, sizeof (w), 1, stream) != 1)
    return EOF;
  return w;
}

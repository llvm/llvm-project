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
#include <string.h>

/* Generate a unique temporary filename using up to five characters of
   PFX if it is not NULL.  The directory to put this file in is
   searched for as follows: First the environment variable "TMPDIR" is
   checked.  If it contains the name of a directory, that directory is
   used.  If not and if DIR is not NULL, that value is checked.  If
   that fails, P_tmpdir is tried and finally "/tmp".  The storage for
   the filename is allocated by `malloc'.  */
char *
tempnam (const char *dir, const char *pfx)
{
  char buf[FILENAME_MAX];

  if (__path_search (buf, FILENAME_MAX, dir, pfx, 1))
    return NULL;

  if (__gen_tempname (buf, 0, 0, __GT_NOCREATE))
    return NULL;

  return __strdup (buf);
}

link_warning (tempnam,
	      "the use of `tempnam' is dangerous, better use `mkstemp'")

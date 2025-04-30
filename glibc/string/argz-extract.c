/* Routines for dealing with '\0' separated arg vectors.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Miles Bader <miles@gnu.org>

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

#include <argz.h>

/* Puts pointers to each string in ARGZ, plus a terminating 0 element, into
   ARGV, which must be large enough to hold them all.  */
void
__argz_extract (const char *argz, size_t len, char **argv)
{
  while (len > 0)
    {
      size_t part_len = strlen (argz);
      *argv++ = (char *) argz;
      argz += part_len + 1;
      len -= part_len + 1;
    }
  *argv = 0;
}
weak_alias (__argz_extract, argz_extract)

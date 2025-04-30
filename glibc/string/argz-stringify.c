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
#include <string.h>

/* Make '\0' separated arg vector ARGZ printable by converting all the '\0's
   except the last into the character SEP.  */
void
__argz_stringify (char *argz, size_t len, int sep)
{
  if (len > 0)
    while (1)
      {
	size_t part_len = __strnlen (argz, len);
	argz += part_len;
	len -= part_len;
	if (len-- <= 1)		/* includes final '\0' we want to stop at */
	  break;
	*argz++ = sep;
      }
}
libc_hidden_def (__argz_stringify)
weak_alias (__argz_stringify, argz_stringify)

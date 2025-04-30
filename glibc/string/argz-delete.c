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
#include <stdlib.h>

/* Delete ENTRY from ARGZ & ARGZ_LEN, if any.  */
void
argz_delete (char **argz, size_t *argz_len, char *entry)
{
  if (entry)
    /* Get rid of the old value for NAME.  */
    {
      size_t entry_len = strlen (entry) + 1;
      *argz_len -= entry_len;
      memmove (entry, entry + entry_len, *argz_len - (entry - *argz));
      if (*argz_len == 0)
	{
	  free (*argz);
	  *argz = 0;
	}
    }
}
libc_hidden_def (argz_delete)

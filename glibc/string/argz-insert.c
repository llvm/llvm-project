/* Routines for dealing with '\0' separated arg vectors.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Miles Bader <miles@gnu.ai.mit.edu>

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

/* Insert ENTRY into ARGZ & ARGZ_LEN before BEFORE, which should be an
   existing entry in ARGZ; if BEFORE is NULL, ENTRY is appended to the end.
   Since ARGZ's first entry is the same as ARGZ, argz_insert (ARGZ, ARGZ_LEN,
   ARGZ, ENTRY) will insert ENTRY at the beginning of ARGZ.  If BEFORE is not
   in ARGZ, EINVAL is returned, else if memory can't be allocated for the new
   ARGZ, ENOMEM is returned, else 0.  */
error_t
__argz_insert (char **argz, size_t *argz_len, char *before, const char *entry)
{
  if (! before)
    return __argz_add (argz, argz_len, entry);

  if (before < *argz || before >= *argz + *argz_len)
    return EINVAL;

  if (before > *argz)
    /* Make sure before is actually the beginning of an entry.  */
    while (before[-1])
      before--;

  {
    size_t after_before = *argz_len - (before - *argz);
    size_t entry_len = strlen  (entry) + 1;
    size_t new_argz_len = *argz_len + entry_len;
    char *new_argz = realloc (*argz, new_argz_len);

    if (new_argz)
      {
	before = new_argz + (before - *argz);
	memmove (before + entry_len, before, after_before);
	memmove (before, entry, entry_len);
	*argz = new_argz;
	*argz_len = new_argz_len;
	return 0;
      }
    else
      return ENOMEM;
  }
}
weak_alias (__argz_insert, argz_insert)

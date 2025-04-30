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
#include <stdlib.h>
#include <string.h>

/* Make a '\0' separated arg vector from a unix argv vector, returning it in
   ARGZ, and the total length in LEN.  If a memory allocation error occurs,
   ENOMEM is returned, otherwise 0.  */
error_t
__argz_create (char *const argv[], char **argz, size_t *len)
{
  int argc;
  size_t tlen = 0;
  char *const *ap;
  char *p;

  for (argc = 0; argv[argc] != NULL; ++argc)
    tlen += strlen (argv[argc]) + 1;

  if (tlen == 0)
    *argz = NULL;
  else
    {
      *argz = malloc (tlen);
      if (*argz == NULL)
	return ENOMEM;

      for (p = *argz, ap = argv; *ap; ++ap, ++p)
	p = __stpcpy (p, *ap);
    }
  *len = tlen;

  return 0;
}
weak_alias (__argz_create, argz_create)

/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>, 1996.

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
#include <errno.h>
#include <stdlib.h>
#include <string.h>


error_t
__argz_create_sep (const char *string, int delim, char **argz, size_t *len)
{
  size_t nlen = strlen (string) + 1;

  if (nlen > 1)
    {
      const char *rp;
      char *wp;

      *argz = (char *) malloc (nlen);
      if (*argz == NULL)
	return ENOMEM;

      rp = string;
      wp = *argz;
      do
	if (*rp == delim)
	  {
	    if (wp > *argz && wp[-1] != '\0')
	      *wp++ = '\0';
	    else
	      --nlen;
	  }
	else
	  *wp++ = *rp;
      while (*rp++ != '\0');

      if (nlen == 0)
	{
	  free (*argz);
	  *argz = NULL;
	  *len = 0;
	}

      *len = nlen;
    }
  else
    {
      *argz = NULL;
      *len = 0;
    }

  return 0;
}
weak_alias (__argz_create_sep, argz_create_sep)

/* Find path of executable.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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

#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <ldsodefs.h>

#include <dl-dst.h>


const char *
_dl_get_origin (void)
{
  char *result = (char *) -1;
  /* We use the environment variable LD_ORIGIN_PATH.  If it is set make
     a copy and strip out trailing slashes.  */
  if (GLRO(dl_origin_path) != NULL)
    {
      size_t len = strlen (GLRO(dl_origin_path));
      result = (char *) malloc (len + 1);
      if (result == NULL)
	result = (char *) -1;
      else
	{
	  char *cp = __mempcpy (result, GLRO(dl_origin_path), len);
	  while (cp > result + 1 && cp[-1] == '/')
	    --cp;
	  *cp = '\0';
	}
    }

  return result;
}

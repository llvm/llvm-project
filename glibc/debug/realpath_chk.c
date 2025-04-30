/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


char *
__realpath_chk (const char *buf, char *resolved, size_t resolvedlen)
{
#ifdef PATH_MAX
  if (resolvedlen < PATH_MAX)
    __chk_fail ();

  return __realpath (buf, resolved);
#else
  long int pathmax =__pathconf (buf, _PC_PATH_MAX);
  if (pathmax != -1)
    {
      /* We do have a fixed limit.  */
      if (resolvedlen < pathmax)
	__chk_fail ();

      return __realpath (buf, resolved);
    }

  /* Since there is no fixed limit we check whether the size is large
     enough.  */
  char *res = __realpath (buf, NULL);
  if (res != NULL)
    {
      size_t actlen = strlen (res) + 1;
      if (actlen > resolvedlen)
	__chk_fail ();

      memcpy (resolved, res, actlen);
      free (res);
      res = resolved;
    }

  return res;
#endif
}

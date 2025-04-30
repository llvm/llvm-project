/* Error-checking replacement for "mkdir -p".
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <support/support.h>
#include <support/check.h>
#include <support/xunistd.h>

#include <stdlib.h>
#include <string.h>
#include <errno.h>

/* Equivalent of "mkdir -p".  Any failures cause FAIL_EXIT1 so no
   return code is needed.  */

void
xmkdirp (const char *path, mode_t mode)
{
  struct stat s;
  const char *slash_p;
  int rv;

  if (path[0] == 0)
    return;

  if (stat (path, &s) == 0)
    {
      if (S_ISDIR (s.st_mode))
	return;
      errno = EEXIST;
      FAIL_EXIT1 ("mkdir_p (\"%s\", 0%o): %m", path, mode);
    }

  slash_p = strrchr (path, '/');
  if (slash_p != NULL)
    {
      while (slash_p > path && slash_p[-1] == '/')
	--slash_p;
      if (slash_p > path)
	{
	  char *parent = xstrndup (path, slash_p - path);
	  xmkdirp (parent, mode);
	  free (parent);
	}
    }

  rv = mkdir (path, mode);
  if (rv != 0)
    FAIL_EXIT1 ("mkdir_p (\"%s\", 0%o): %m", path, mode);

  return;
}

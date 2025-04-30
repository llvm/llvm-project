/* dirname - return directory part of PATH.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

#include <libgen.h>
#include <string.h>


char *
dirname (char *path)
{
  static const char dot[] = ".";
  char *last_slash;

  /* Find last '/'.  */
  last_slash = path != NULL ? strrchr (path, '/') : NULL;

  if (last_slash != NULL && last_slash != path && last_slash[1] == '\0')
    {
      /* Determine whether all remaining characters are slashes.  */
      char *runp;

      for (runp = last_slash; runp != path; --runp)
	if (runp[-1] != '/')
	  break;

      /* The '/' is the last character, we have to look further.  */
      if (runp != path)
	last_slash = __memrchr (path, '/', runp - path);
    }

  if (last_slash != NULL)
    {
      /* Determine whether all remaining characters are slashes.  */
      char *runp;

      for (runp = last_slash; runp != path; --runp)
	if (runp[-1] != '/')
	  break;

      /* Terminate the path.  */
      if (runp == path)
	{
	  /* The last slash is the first character in the string.  We have to
	     return "/".  As a special case we have to return "//" if there
	     are exactly two slashes at the beginning of the string.  See
	     XBD 4.10 Path Name Resolution for more information.  */
	  if (last_slash == path + 1)
	    ++last_slash;
	  else
	    last_slash = path + 1;
	}
      else
	last_slash = runp;

      last_slash[0] = '\0';
    }
  else
    /* This assignment is ill-designed but the XPG specs require to
       return a string containing "." in any case no directory part is
       found and so a static and constant string is required.  */
    path = (char *) dot;

  return path;
}

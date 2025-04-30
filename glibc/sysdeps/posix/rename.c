/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <unistd.h>
#include <errno.h>

/* Rename the file OLD to NEW.  */
int
rename (const char *old, const char *new)
{
  int save = errno;
  if (__link (old, new) < 0)
    {
      if (errno == EEXIST)
	{
	  __set_errno (save);
	  /* Race condition, required for 1003.1 conformance.  */
	  if (__unlink (new) < 0
	      || __link (old, new) < 0)
	    return -1;
	}
      else
	return -1;
    }
  if (__unlink (old) < 0)
    {
      save = errno;
      if (__unlink (new) == 0)
	__set_errno (save);
      return -1;
    }
  return 0;
}

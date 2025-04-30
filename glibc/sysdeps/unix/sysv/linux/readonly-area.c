/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <stdlib.h>
#include <string.h>
#include "libio/libioP.h"

/* Return 1 if the whole area PTR .. PTR+SIZE is not writable.
   Return -1 if it is writable.  */

int
__readonly_area (const char *ptr, size_t size)
{
  const void *ptr_end = ptr + size;

  FILE *fp = fopen ("/proc/self/maps", "rce");
  if (fp == NULL)
    {
      /* It is the system administrator's choice to not have /proc
	 available to this process (e.g., because it runs in a chroot
	 environment.  Don't fail in this case.  */
      if (errno == ENOENT
	  /* The kernel has a bug in that a process is denied access
	     to the /proc filesystem if it is set[ug]id.  There has
	     been no willingness to change this in the kernel so
	     far.  */
	  || errno == EACCES)
	return 1;
      return -1;
    }

  /* We need no locking.  */
  __fsetlocking (fp, FSETLOCKING_BYCALLER);

  char *line = NULL;
  size_t linelen = 0;

  while (! __feof_unlocked (fp))
    {
      if (__getdelim (&line, &linelen, '\n', fp) <= 0)
	break;

      char *p;
      uintptr_t from = strtoul (line, &p, 16);

      if (p == line || *p++ != '-')
	break;

      char *q;
      uintptr_t to = strtoul (p, &q, 16);

      if (q == p || *q++ != ' ')
	break;

      if (from < (uintptr_t) ptr_end && to > (uintptr_t) ptr)
	{
	  /* Found an entry that at least partially covers the area.  */
	  if (*q++ != 'r' || *q++ != '-')
	    break;

	  if (from <= (uintptr_t) ptr && to >= (uintptr_t) ptr_end)
	    {
	      size = 0;
	      break;
	    }
	  else if (from <= (uintptr_t) ptr)
	    size -= to - (uintptr_t) ptr;
	  else if (to >= (uintptr_t) ptr_end)
	    size -= (uintptr_t) ptr_end - from;
	  else
	    size -= to - from;

	  if (!size)
	    break;
	}
    }

  fclose (fp);
  free (line);

  /* If the whole area between ptr and ptr_end is covered by read-only
     VMAs, return 1.  Otherwise return -1.  */
  return size == 0 ? 1 : -1;
}

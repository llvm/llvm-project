/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Mark Kettenis <kettenis@phys.uva.nl>, 1997.

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

#include <libc-lock.h>
#include <stdlib.h>
#include <string.h>
#include <utmp.h>

#include "utmp-private.h"


/* This is the default name.  */
static const char default_file_name[] = _PATH_UTMP;

/* Current file name.  */
const char *__libc_utmp_file_name = (const char *) default_file_name;

/* We have to use the lock in getutent_r.c.  */
__libc_lock_define (extern, __libc_utmp_lock attribute_hidden)


int
__utmpname (const char *file)
{
  int result = -1;

  __libc_lock_lock (__libc_utmp_lock);

  /* Close the old file.  */
  __libc_endutent ();

  if (strcmp (file, __libc_utmp_file_name) != 0)
    {
      if (strcmp (file, default_file_name) == 0)
	{
	  free ((char *) __libc_utmp_file_name);

	  __libc_utmp_file_name = default_file_name;
	}
      else
	{
	  char *file_name = __strdup (file);
	  if (file_name == NULL)
	    /* Out of memory.  */
	    goto done;

	  if (__libc_utmp_file_name != default_file_name)
	    free ((char *) __libc_utmp_file_name);

	  __libc_utmp_file_name = file_name;
	}
    }

  result = 0;

done:
  __libc_lock_unlock (__libc_utmp_lock);
  return result;
}
libc_hidden_def (__utmpname)
weak_alias (__utmpname, utmpname)

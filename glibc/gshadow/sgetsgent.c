/* Copyright (C) 2009-2021 Free Software Foundation, Inc.
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
#include <libc-lock.h>
#include <gshadow.h>
#include <stdlib.h>


/* A reasonable size for a buffer to start with.  */
#define BUFLEN_SPWD 1024

/* We need to protect the dynamic buffer handling.  */
__libc_lock_define_initialized (static, lock);

/* Read one shadow entry from the given stream.  */
struct sgrp *
sgetsgent (const char *string)
{
  static char *buffer;
  static size_t buffer_size;
  static struct sgrp resbuf;
  struct sgrp *result;
  int save;

  /* Get lock.  */
  __libc_lock_lock (lock);

  /* Allocate buffer if not yet available.  */
  if (buffer == NULL)
    {
      buffer_size = BUFLEN_SPWD;
      buffer = malloc (buffer_size);
    }

  while (buffer != NULL
	 && __sgetsgent_r (string, &resbuf, buffer, buffer_size, &result) != 0
	 && errno == ERANGE)
    {
      char *new_buf;
      buffer_size += BUFLEN_SPWD;
      new_buf = realloc (buffer, buffer_size);
      if (new_buf == NULL)
	{
	  /* We are out of memory.  Free the current buffer so that the
	     process gets a chance for a normal termination.  */
	  save = errno;
	  free (buffer);
	  __set_errno (save);
	}
      buffer = new_buf;
    }

  if (buffer == NULL)
    result = NULL;

  /* Release lock.  Preserve error value.  */
  save = errno;
  __libc_lock_unlock (lock);
  __set_errno (save);

  return result;
}

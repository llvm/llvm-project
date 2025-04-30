/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
#include <stdlib.h>
#include <netdb.h>
#include "nsswitch.h"

void *
__nss_getent (getent_r_function func, void **resbuf, char **buffer,
	      size_t buflen, size_t *buffer_size, int *h_errnop)
{
  void *result;

  if (*buffer == NULL)
    {
      *buffer_size = buflen;
      *buffer = malloc (*buffer_size);
    }

  while (*buffer != NULL
	 && func (resbuf, *buffer, *buffer_size, &result, h_errnop) == ERANGE
	 && (h_errnop == NULL || *h_errnop == NETDB_INTERNAL))
    {
      char *new_buf;
      *buffer_size *= 2;
      new_buf = realloc (*buffer, *buffer_size);
      if (new_buf == NULL)
	{
	  /* We are out of memory.  Free the current buffer so that the
	     process gets a chance for a normal termination.  */
	  int save = errno;
	  free (*buffer);
	  __set_errno (save);
	}
      *buffer = new_buf;
    }

  if (*buffer == NULL)
    result = NULL;

  return result;
}

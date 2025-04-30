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

#include <errno.h>
#include <limits.h>
#include <termios.h>
#include <stdlib.h>

#include "ttyname.h"

static char *ttyname_buf = NULL;

libc_freeres_fn (free_mem)
{
  free (ttyname_buf);
}

/* Return the pathname of the terminal FD is open on, or NULL on errors.
   The returned storage is good only until the next call to this function.  */
char *
ttyname (int fd)
{
  /* isatty check, tcgetattr is used because it sets the correct
     errno (EBADF resp. ENOTTY) on error.  Fast error path to avoid the
     allocation  */
  struct termios term;
  if (__glibc_unlikely (__tcgetattr (fd, &term) < 0))
    return NULL;

  if (ttyname_buf == NULL)
    {
      ttyname_buf = malloc (PATH_MAX);
      if (ttyname_buf == NULL)
	return NULL;
    }

  int result = __ttyname_r (fd, ttyname_buf, PATH_MAX);
  if (result != 0)
    {
      __set_errno (result);
      return NULL;
    }
  return ttyname_buf;
}

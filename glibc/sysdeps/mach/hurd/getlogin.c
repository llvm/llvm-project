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

#include <stddef.h>
#include <errno.h>
#include <unistd.h>
#include <hurd.h>

/* Return the login name of the user, or NULL if it can't be determined.
   The returned pointer, if not NULL, is good only until the next call.  */
char *
getlogin (void)
{
  static char login[1024];	/* XXX */
  error_t err;

  if (err = __USEPORT (PROC, __proc_getlogin (port, login)))
    {
      errno = err;
      return NULL;
    }

  return login;
}

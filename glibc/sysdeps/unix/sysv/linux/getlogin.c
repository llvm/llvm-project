/* Copyright (C) 2010-2021 Free Software Foundation, Inc.
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

#include <pwd.h>
#include <unistd.h>
#include <not-cancel.h>

#define STATIC static
#define getlogin getlogin_fd0
#include <sysdeps/unix/getlogin.c>
#undef getlogin


/* Return the login name of the user, or NULL if it can't be determined.
   The returned pointer, if not NULL, is good only until the next call.  */

char *
getlogin (void)
{
  int res = __getlogin_r_loginuid (name, sizeof (name));
  if (res >= 0)
    return res == 0 ? name : NULL;

  return getlogin_fd0 ();
}

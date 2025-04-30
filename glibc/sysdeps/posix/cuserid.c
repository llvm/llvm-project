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

#include <pwd.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>

/* Return the username of the caller.
   If S is not NULL, it points to a buffer of at least L_cuserid bytes
   into which the name is copied; otherwise, a static buffer is used.  */
char *
cuserid (char *s)
{
  static char name[L_cuserid];
  char buf[NSS_BUFLEN_PASSWD];
  struct passwd pwent;
  struct passwd *pwptr;

  if (__getpwuid_r (__geteuid (), &pwent, buf, sizeof (buf), &pwptr)
      || pwptr == NULL)
    {
      if (s != NULL)
	s[0] = '\0';
      return s;
    }

  if (s == NULL)
    s = name;
  s[L_cuserid - 1] = '\0';
  return strncpy (s, pwptr->pw_name, L_cuserid - 1);
}

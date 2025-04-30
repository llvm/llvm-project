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

#include <alloca.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <pwd.h>


/* Re-construct the password-file line for the given uid
   in the given buffer.  This knows the format that the caller
   will expect, but this need not be the format of the password file.  */

int __getpw (__uid_t uid, char *buf);

int
__getpw (__uid_t uid, char *buf)
{
  size_t buflen;
  char *tmpbuf;
  struct passwd resbuf, *p;

  if (buf == NULL)
    {
      __set_errno (EINVAL);
      return -1;
    }

  buflen = __sysconf (_SC_GETPW_R_SIZE_MAX);
  tmpbuf = alloca (buflen);

  if (__getpwuid_r (uid, &resbuf, tmpbuf, buflen, &p) != 0)
    return -1;

  if (p == NULL)
    return -1;

  if (sprintf (buf, "%s:%s:%lu:%lu:%s:%s:%s", p->pw_name, p->pw_passwd,
	       (unsigned long int) p->pw_uid, (unsigned long int) p->pw_gid,
	       p->pw_gecos, p->pw_dir, p->pw_shell) < 0)
    return -1;

  return 0;
}
weak_alias (__getpw, getpw)

link_warning (getpw, "the `getpw' function is dangerous and should not be used.")

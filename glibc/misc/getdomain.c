/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

/* Put the name of the current YP domain in no more than LEN bytes of NAME.
   The result is null-terminated if LEN is large enough for the full
   name and the terminator.  */

#include <errno.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/utsname.h>
#include <string.h>

#if _UTSNAME_DOMAIN_LENGTH
/* The `uname' information includes the domain name.  */

int
getdomainname (char *name, size_t len)
{
  struct utsname u;
  size_t u_len;

  if (uname (&u) < 0)
    return -1;

  u_len = strlen (u.domainname);
  memcpy (name, u.domainname, MIN (u_len + 1, len));
  return 0;
}

#else

int
getdomainname (char *name, size_t len)
{
  __set_errno (ENOSYS);
  return -1;
}

stub_warning (getdomainname)

#endif

libc_hidden_def (getdomainname)

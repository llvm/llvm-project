/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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
#include <string.h>
#include <unistd.h>
#include <sys/utsname.h>

/* Put the name of the current host in no more than LEN bytes of NAME.
   The result is null-terminated if LEN is large enough for the full
   name and the terminator.  */
int
__gethostname (char *name, size_t len)
{
  struct utsname buf;
  size_t node_len;

  if (__uname (&buf))
    return -1;

  node_len = strlen (buf.nodename) + 1;
  memcpy (name, buf.nodename, len < node_len ? len : node_len);

  if (node_len > len)
    {
      __set_errno (ENAMETOOLONG);
      return -1;
    }
  return 0;
}

weak_alias (__gethostname, gethostname)

/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#include <unistd.h>
#include "hurdhost.h"

/* Set the name of the current YP domain to NAME, which is LEN bytes long.
   This call is restricted to the super-user.  */
int
setdomainname (const char *name, size_t len)
{
  /* The NIS domain name is just the contents of the file /etc/nisdomain.  */
  ssize_t n = _hurd_set_host_config ("/etc/nisdomain", name, len);
  return n < 0 ? -1 : 0;
}

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

#include <unistd.h>
#include <hurd.h>
#include "hurdhost.h"

/* Return the current machine's Internet number.  */
long int
gethostid (void)
{
  /* The hostid is just the contents of the file /etc/hostid,
     kept as text of hexadecimal digits.  */
  /* XXX this is supposed to come from the hardware serial number */
  char buf[8];
  ssize_t n = _hurd_get_host_config ("/etc/hostid", buf, sizeof buf);
  if (n < 0)
    return -1;
  return strtol (buf, NULL, 16);
}

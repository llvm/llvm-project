/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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
#include <_itoa.h>

/* Set the current machine's Internet number to ID.
   This call is restricted to the super-user.  */
/* XXX should be __sethostid?
       isn't hostid supposed to be hardwired and unchangeable? */
int
sethostid (long int id)
{
  char buf[8], *bp;
  ssize_t n;

  /* The hostid is kept in the file /etc/hostid,
     eight characters of upper-case hexadecimal.  */

  bp = _itoa_word (id, &buf[sizeof buf], 16, 1);
  while (bp > buf)
    *--bp = '0';

  n = _hurd_set_host_config ("/etc/hostid", buf, sizeof buf);
  return n < 0 ? -1 : 0;
}

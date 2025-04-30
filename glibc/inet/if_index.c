/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <net/if.h>
#include <errno.h>
#include <stddef.h>

unsigned int
__if_nametoindex (const char *ifname)
{
  __set_errno (ENOSYS);
  return 0;
}
libc_hidden_def (__if_nametoindex)
weak_alias (__if_nametoindex, if_nametoindex)
libc_hidden_weak (if_nametoindex)
stub_warning (if_nametoindex)

char *
__if_indextoname (unsigned int ifindex, char ifname[IF_NAMESIZE])
{
  __set_errno (ENOSYS);
  return NULL;
}
weak_alias (__if_indextoname, if_indextoname)
libc_hidden_weak (if_indextoname)
stub_warning (if_indextoname)

void
__if_freenameindex (struct if_nameindex *ifn)
{
}
libc_hidden_def (__if_freenameindex)
weak_alias (__if_freenameindex, if_freenameindex)
libc_hidden_weak (if_freenameindex)
stub_warning (if_freenameindex)

struct if_nameindex *
__if_nameindex (void)
{
  __set_errno (ENOSYS);
  return NULL;
}
weak_alias (__if_nameindex, if_nameindex)
libc_hidden_weak (if_nameindex)
stub_warning (if_nameindex)

/* getsysstats - Determine various system internal values, stub version.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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
#include <sys/sysinfo.h>

int
__get_nprocs_conf (void)
{
  /* We don't know how to determine the number.  Simply return always 1.  */
  return 1;
}
libc_hidden_def (__get_nprocs_conf)
weak_alias (__get_nprocs_conf, get_nprocs_conf)

link_warning (get_nprocs_conf, "warning: get_nprocs_conf will always return 1")



int
__get_nprocs (void)
{
  /* We don't know how to determine the number.  Simply return always 1.  */
  return 1;
}
libc_hidden_def (__get_nprocs)
weak_alias (__get_nprocs, get_nprocs)

link_warning (get_nprocs, "warning: get_nprocs will always return 1")


long int
__get_phys_pages (void)
{
  /* We have no general way to determine this value.  */
  __set_errno (ENOSYS);
  return -1;
}
libc_hidden_def (__get_phys_pages)
weak_alias (__get_phys_pages, get_phys_pages)

stub_warning (get_phys_pages)


long int
__get_avphys_pages (void)
{
  /* We have no general way to determine this value.  */
  __set_errno (ENOSYS);
  return -1;
}
libc_hidden_def (__get_avphys_pages)
weak_alias (__get_avphys_pages, get_avphys_pages)

stub_warning (get_avphys_pages)

/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
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
#include <unistd.h>


int
__getgroups_chk (int size, __gid_t list[], size_t listlen)
{
  if (__glibc_unlikely (size < 0))
    {
      __set_errno (EINVAL);
      return -1;
    }

  if (__glibc_unlikely (size * sizeof (__gid_t) > listlen))
    __chk_fail ();

  return __getgroups (size, list);
}

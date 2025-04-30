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

#include <errno.h>
#include <stddef.h>
#include <unistd.h>
#include <hurd.h>
#include <hurd/fd.h>

/* Change the owner and group of the file referred to by FD.  */
int
__fchown (int fd, uid_t owner, gid_t group)
{
  error_t err;

  if (err = HURD_DPORT_USE (fd, __file_chown (port, owner, group)))
    return __hurd_dfail (fd, err);

  return 0;
}

weak_alias (__fchown, fchown)

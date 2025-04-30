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
#include <sys/types.h>
#include <unistd.h>
#include <hurd.h>
#include <hurd/fd.h>

/* Truncate the file referenced by FD to LENGTH bytes.  */
int
__ftruncate (int fd, __off_t length)
{
  error_t err;
  if (err = HURD_DPORT_USE (fd, __file_set_size (port, length)))
    return __hurd_dfail (fd, err);
  return 0;
}

weak_alias (__ftruncate, ftruncate)

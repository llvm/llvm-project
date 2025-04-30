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

#include <unistd.h>

/* Create a one-way communication channel (pipe).
   Actually the channel is two-way on the Hurd.
   If successful, two file descriptors are stored in FDS;
   bytes written on FDS[1] can be read from FDS[0].
   Returns 0 if successful, -1 if not.  */
int
__pipe (int fds[2])
{
  return __pipe2 (fds, 0);
}
libc_hidden_def (__pipe)
weak_alias (__pipe, pipe)

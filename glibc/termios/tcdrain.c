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
#include <termios.h>

/* Wait for pending output to be written on FD.  */
int
__libc_tcdrain (int fd)
{
  if (fd < 0)
    {
      __set_errno (EBADF);
      return -1;
    }

  __set_errno (ENOSYS);
  return -1;
}
weak_alias (__libc_tcdrain, tcdrain)


stub_warning (tcdrain)
